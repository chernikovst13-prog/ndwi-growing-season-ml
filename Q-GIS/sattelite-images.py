import numpy as np
from osgeo import gdal, ogr, osr
import os
import re
import csv
from datetime import datetime

try:
    from qgis.core import QgsProject, QgsRasterLayer
    HAS_QGIS = True
    print("QGIS API доступен")
except Exception:
    HAS_QGIS = False
    print("QGIS API НЕ доступен")


def _robust_stats(array):
    valid = np.isfinite(array)
    values = array[valid].astype(np.float32)
    if values.size == 0:
        raise RuntimeError("No valid pixels in input raster")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        mad = float(np.std(values))
    if mad == 0.0:
        mad = 1.0

    return median, mad


def _robust_z(array):
    median, mad = _robust_stats(array)
    return (array - median) / (mad + 1e-6)


def _night_cloud_mask_northern(
    b1, b2, b3, b4, b5, b6,
    cold_k=0.75, texture_k=0.5, ratio_k=0.6,
    min_votes=4, dark_min_votes=3, dark_seed_votes=4,
    bright_min_votes=3, bright_seed_votes=4, min_neighbors=0,
):
    eps = 1e-6
    temp_mean = (b5 + b6) / 2.0
    temp_gap = np.abs(b5 - b6)

    r34 = b3 / (b4 + eps)
    r56 = b5 / (b6 + eps)
    r12_3 = (b1 + b2) / (b3 + eps)

    z_b1 = _robust_z(b1)
    z_b2 = _robust_z(b2)
    z_b5 = _robust_z(b5)
    z_b6 = _robust_z(b6)
    z_temp = _robust_z(temp_mean)
    z_gap = _robust_z(temp_gap)
    z_r34 = _robust_z(r34)
    z_r56 = _robust_z(r56)
    z_r12_3 = _robust_z(r12_3)

    texture = (
        np.abs(temp_mean - np.roll(temp_mean, 1, axis=0)) +
        np.abs(temp_mean - np.roll(temp_mean, -1, axis=0)) +
        np.abs(temp_mean - np.roll(temp_mean, 1, axis=1)) +
        np.abs(temp_mean - np.roll(temp_mean, -1, axis=1))
    ) / 4.0
    z_texture = _robust_z(texture)

    votes = (
        (z_b5 < (-0.4 - 0.6 * cold_k)).astype(np.uint8) +
        (z_b6 < (-0.4 - 0.6 * cold_k)).astype(np.uint8) +
        (z_temp < (-0.45 - 0.7 * cold_k)).astype(np.uint8) +
        (z_r34 > (0.1 + 0.5 * ratio_k)).astype(np.uint8) +
        (z_r56 < (0.0 - 0.35 * ratio_k)).astype(np.uint8) +
        (z_r12_3 < (0.05 - 0.5 * ratio_k)).astype(np.uint8) +
        (z_texture > (-1.0 - 0.5 * texture_k)).astype(np.uint8) +
        (z_gap < (0.8 + 0.4 * texture_k)).astype(np.uint8)
    )

    dark_votes = (
        (z_b1 < (0.15 - 0.5 * ratio_k)).astype(np.uint8) +
        (z_b2 < (0.15 - 0.5 * ratio_k)).astype(np.uint8) +
        (z_temp < (-0.6 - 0.8 * cold_k)).astype(np.uint8) +
        (z_r34 > (-0.1 + 0.45 * ratio_k)).astype(np.uint8) +
        (z_r12_3 < (-0.15 - 0.55 * ratio_k)).astype(np.uint8) +
        (z_texture > (-1.25 - 0.6 * texture_k)).astype(np.uint8)
    )

    bright_votes = (
        (z_b1 > (0.55 + 0.45 * ratio_k)).astype(np.uint8) +
        (z_b2 > (0.55 + 0.45 * ratio_k)).astype(np.uint8) +
        (z_r12_3 > (0.25 + 0.35 * ratio_k)).astype(np.uint8) +
        (z_r34 > (-0.15 + 0.3 * ratio_k)).astype(np.uint8) +
        (z_gap < (1.1 + 0.2 * texture_k)).astype(np.uint8) +
        (z_texture > (-0.9 - 0.4 * texture_k)).astype(np.uint8)
    )

    cloud_core = votes >= min_votes
    dark_seed = dark_votes >= max(dark_seed_votes, dark_min_votes)
    dark_core = dark_votes >= dark_min_votes
    bright_seed = bright_votes >= max(bright_seed_votes, bright_min_votes)
    bright_core = bright_votes >= bright_min_votes

    cloud_soft = votes >= max(2, min_votes - 1)
    dark_soft = dark_votes >= max(2, dark_min_votes - 1)
    very_dark_context = (
        (z_temp < (-0.2 - 0.6 * cold_k)) &
        (z_r34 > (-0.2 + 0.35 * ratio_k)) &
        (z_texture > -1.6)
    )

    core_neighbors = (
        np.roll(cloud_core, 1, axis=0).astype(np.uint8) +
        np.roll(cloud_core, -1, axis=0).astype(np.uint8) +
        np.roll(cloud_core, 1, axis=1).astype(np.uint8) +
        np.roll(cloud_core, -1, axis=1).astype(np.uint8)
    )
    dark_neighbors = (
        np.roll(dark_seed, 1, axis=0).astype(np.uint8) +
        np.roll(dark_seed, -1, axis=0).astype(np.uint8) +
        np.roll(dark_seed, 1, axis=1).astype(np.uint8) +
        np.roll(dark_seed, -1, axis=1).astype(np.uint8)
    )
    bright_neighbors = (
        np.roll(bright_seed, 1, axis=0).astype(np.uint8) +
        np.roll(bright_seed, -1, axis=0).astype(np.uint8) +
        np.roll(bright_seed, 1, axis=1).astype(np.uint8) +
        np.roll(bright_seed, -1, axis=1).astype(np.uint8)
    )

    cloud_mask = (
        cloud_core |
        (cloud_soft & (core_neighbors >= 2)) |
        dark_seed |
        (dark_core & (dark_neighbors >= 1)) |
        bright_seed |
        (bright_core & (bright_neighbors >= 1))
    )

    seed_for_dark_soft = cloud_core | dark_seed
    seed_neighbors = (
        np.roll(seed_for_dark_soft, 1, axis=0).astype(np.uint8) +
        np.roll(seed_for_dark_soft, -1, axis=0).astype(np.uint8) +
        np.roll(seed_for_dark_soft, 1, axis=1).astype(np.uint8) +
        np.roll(seed_for_dark_soft, -1, axis=1).astype(np.uint8)
    )
    cloud_mask = cloud_mask | (dark_soft & (seed_neighbors >= 2))

    dark_region = dark_seed | dark_soft
    for _ in range(2):
        dark_region_neighbors = (
            np.roll(dark_region, 1, axis=0).astype(np.uint8) +
            np.roll(dark_region, -1, axis=0).astype(np.uint8) +
            np.roll(dark_region, 1, axis=1).astype(np.uint8) +
            np.roll(dark_region, -1, axis=1).astype(np.uint8) +
            np.roll(np.roll(dark_region, 1, axis=0), 1, axis=1).astype(np.uint8) +
            np.roll(np.roll(dark_region, 1, axis=0), -1, axis=1).astype(np.uint8) +
            np.roll(np.roll(dark_region, -1, axis=0), 1, axis=1).astype(np.uint8) +
            np.roll(np.roll(dark_region, -1, axis=0), -1, axis=1).astype(np.uint8)
        )
        dark_region = dark_region | (very_dark_context & (dark_region_neighbors >= 1))
    cloud_mask = cloud_mask | dark_region

    non_land_guard = (z_temp < 0.4) & ((z_b5 < 0.7) | (z_b6 < 0.7) | (z_gap < 0.8))
    land_like = (
        ((z_temp > 0.45) & (z_texture < 0.15) & (z_gap < 0.45)) |
        ((z_b5 > 0.85) & (z_b6 > 0.85) & (z_texture < 0.3))
    )
    land_votes = (
        (z_temp > 0.15).astype(np.uint8) +
        (z_b5 > 0.65).astype(np.uint8) +
        (z_b6 > 0.65).astype(np.uint8) +
        (z_gap < 0.35).astype(np.uint8) +
        (z_texture < 0.05).astype(np.uint8)
    )
    land_suspect = land_votes >= 4

    stable_thermal_land = (z_temp > -0.05) & (z_gap < 0.28) & (z_texture < 0.1)
    strict_non_land = (
        (z_temp < 0.35) &
        ((z_b5 < 0.65) | (z_b6 < 0.65) | (z_gap < 0.75))
    )

    cloud_mask = cloud_mask & non_land_guard & strict_non_land & (~land_like) & (~land_suspect) & (~stable_thermal_land)

    neighbor_count = (
        np.roll(cloud_mask, 1, axis=0).astype(np.uint8) +
        np.roll(cloud_mask, -1, axis=0).astype(np.uint8) +
        np.roll(cloud_mask, 1, axis=1).astype(np.uint8) +
        np.roll(cloud_mask, -1, axis=1).astype(np.uint8)
    )
    return cloud_mask & (neighbor_count >= min_neighbors)


def _get_raster_bounds(ds):
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + gt[1] * cols + gt[2] * rows
    y_min = y_max + gt[4] * cols + gt[5] * rows
    return x_min, y_min, x_max, y_max


def _read_shrub_mask_aligned(vegetation_raster_path, reference_ds):
    """Читает и выравнивает маску растительности. Возвращает маску в памяти."""
    vegetation_ds = gdal.Open(vegetation_raster_path, gdal.GA_ReadOnly)
    if vegetation_ds is None:
        raise RuntimeError(f"Cannot open vegetation raster: {vegetation_raster_path}")

    if vegetation_ds.RasterCount < 1:
        raise RuntimeError(f"Vegetation raster has no bands: {vegetation_raster_path}")

    ref_cols = reference_ds.RasterXSize
    ref_rows = reference_ds.RasterYSize
    ref_projection = reference_ds.GetProjection()
    ref_bounds = _get_raster_bounds(reference_ds)
    veg_band = vegetation_ds.GetRasterBand(1)
    veg_nodata = veg_band.GetNoDataValue()

    warp_kwargs = dict(
        format="MEM",
        dstSRS=ref_projection,
        outputBounds=ref_bounds,
        width=ref_cols,
        height=ref_rows,
        resampleAlg=gdal.GRA_NearestNeighbour,
        multithread=True,
    )
    if veg_nodata is not None:
        warp_kwargs["srcNodata"] = veg_nodata
        warp_kwargs["dstNodata"] = veg_nodata

    warped_ds = gdal.Warp("", vegetation_ds, **warp_kwargs)
    if warped_ds is None:
        raise RuntimeError(f"Cannot align vegetation raster to reference grid: {vegetation_raster_path}")

    vegetation_array = warped_ds.GetRasterBand(1).ReadAsArray()
    shrub_values = np.array([11, 12, 14], dtype=np.int32)
    shrub_mask = np.isin(vegetation_array, shrub_values)
    
    warped_ds = None
    vegetation_ds = None
    
    return shrub_mask


def _get_safe_output_path(output_path):
    """
    ВСЕГДА возвращает путь к НЕсуществующему файлу.
    Если файл существует и доступен — удаляет его.
    Если файл существует и заблокирован — создаёт новое имя с суффиксом _v2, _v3...
    """
    if not os.path.exists(output_path):
        return output_path
    
    try:
        os.remove(output_path)
        return output_path
    except:
        pass
    
    base, ext = os.path.splitext(output_path)
    for counter in range(2, 1000):
        new_path = f"{base}_v{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        try:
            os.remove(new_path)
            return new_path
        except:
            continue
    
    raise RuntimeError(f"Не удалось найти свободное имя для: {output_path}")


def _write_masked_raster(input_raster_path, output_raster_path, keep_mask):
    """Сохраняет обрезанный снимок с гарантированно уникальным именем."""
    actual_path = _get_safe_output_path(output_raster_path)
    
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {input_raster_path}")

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band_count = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    if keep_mask.shape != (rows, cols):
        raise RuntimeError("Mask shape does not match input raster shape")

    mask_rows, mask_cols = np.where(keep_mask)
    if mask_rows.size == 0:
        raise RuntimeError("No shrub pixels remain after removing clouds")

    row_min = int(mask_rows.min())
    row_max = int(mask_rows.max())
    col_min = int(mask_cols.min())
    col_max = int(mask_cols.max())
    crop_cols = col_max - col_min + 1
    crop_rows = row_max - row_min + 1
    new_geotransform = (
        geotransform[0] + col_min * geotransform[1] + row_min * geotransform[2],
        geotransform[1], geotransform[2],
        geotransform[3] + col_min * geotransform[4] + row_min * geotransform[5],
        geotransform[4], geotransform[5],
    )

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        actual_path, crop_cols, crop_rows, band_count,
        ds.GetRasterBand(1).DataType,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if out_ds is None:
        raise RuntimeError(f"Cannot create output raster: {actual_path}")

    out_ds.SetGeoTransform(new_geotransform)
    out_ds.SetProjection(projection)

    for band_index in range(1, band_count + 1):
        in_band = ds.GetRasterBand(band_index)
        band_array = in_band.ReadAsArray(col_min, row_min, crop_cols, crop_rows)
        crop_mask = keep_mask[row_min: row_max + 1, col_min: col_max + 1]
        nodata_value = in_band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = 0
        masked_array = np.where(crop_mask, band_array, nodata_value)
        out_band = out_ds.GetRasterBand(band_index)
        out_band.WriteArray(masked_array)
        out_band.SetNoDataValue(nodata_value)
        out_band.FlushCache()

    out_ds.FlushCache()
    out_ds = None
    ds = None

    print(f"  ✓ Обрезанный снимок сохранён: {os.path.basename(actual_path)}")


def _rasterize_vector_to_mask(vector_path, reference_ds, burn_value=1):
    """Растеризует векторный слой в маску."""
    vec_ds = ogr.Open(vector_path)
    if vec_ds is None:
        raise RuntimeError(f"Не удалось открыть вектор: {vector_path}")
    
    layer = vec_ds.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"В файле нет слоёв: {vector_path}")
    
    cols = reference_ds.RasterXSize
    rows = reference_ds.RasterYSize
    geotransform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    
    driver = gdal.GetDriverByName("MEM")
    temp_ds = driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    temp_ds.SetGeoTransform(geotransform)
    temp_ds.SetProjection(projection)
    temp_band = temp_ds.GetRasterBand(1)
    temp_band.Fill(0)
    temp_band.SetNoDataValue(0)
    
    gdal.RasterizeLayer(temp_ds, [1], layer, burn_values=[burn_value])
    
    mask_array = temp_band.ReadAsArray().astype(bool)
    
    temp_ds = None
    vec_ds = None
    
    return mask_array


def _calculate_ndwi_mean_in_buffer(input_raster_path, buffer_mask, shrub_mask=None):
    """Рассчитывает средний NDWI в пределах буфера и маски кустарников."""
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster for NDWI: {input_raster_path}")

    if ds.RasterCount < 3:
        raise RuntimeError(f"Expected at least 3 bands for NDWI, found: {ds.RasterCount}")

    b2_band = ds.GetRasterBand(2)
    b3_band = ds.GetRasterBand(3)
    b2 = b2_band.ReadAsArray().astype(np.float32)
    b3 = b3_band.ReadAsArray().astype(np.float32)

    b2_nodata = b2_band.GetNoDataValue()
    b3_nodata = b3_band.GetNoDataValue()

    valid = np.isfinite(b2) & np.isfinite(b3)
    if b2_nodata is not None:
        valid &= b2 != b2_nodata
    if b3_nodata is not None:
        valid &= b3 != b3_nodata

    denominator = b2 + b3
    valid &= np.abs(denominator) > 1e-6
    
    if buffer_mask.shape != b2.shape:
        raise RuntimeError("Buffer mask shape does not match raster shape")
    valid &= buffer_mask
    
    if shrub_mask is not None:
        if shrub_mask.shape != b2.shape:
            raise RuntimeError("Shrub mask shape does not match raster shape")
        valid &= shrub_mask

    if not np.any(valid):
        return None, 0

    ndwi = (b2 - b3) / denominator
    mean_ndwi = float(np.mean(ndwi[valid]))
    pixel_count = int(np.sum(valid))
    
    ds = None
    return mean_ndwi, pixel_count


def _write_ndwi_raster_in_buffer(input_raster_path, output_ndwi_path, buffer_mask, shrub_mask=None):
    """Создаёт растр NDWI с гарантированно уникальным именем."""
    actual_path = _get_safe_output_path(output_ndwi_path)
    
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster for NDWI layer: {input_raster_path}")

    if ds.RasterCount < 3:
        raise RuntimeError(f"Expected at least 3 bands for NDWI, found: {ds.RasterCount}")

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    b2_band = ds.GetRasterBand(2)
    b3_band = ds.GetRasterBand(3)
    b2 = b2_band.ReadAsArray().astype(np.float32)
    b3 = b3_band.ReadAsArray().astype(np.float32)
    b2_nodata = b2_band.GetNoDataValue()
    b3_nodata = b3_band.GetNoDataValue()

    valid = np.isfinite(b2) & np.isfinite(b3)
    if b2_nodata is not None:
        valid &= b2 != b2_nodata
    if b3_nodata is not None:
        valid &= b3 != b3_nodata

    denominator = b2 + b3
    valid &= np.abs(denominator) > 1e-6
    
    valid &= buffer_mask
    
    if shrub_mask is not None:
        valid &= shrub_mask

    ndwi_nodata = np.float32(-9999.0)
    ndwi = np.full((rows, cols), ndwi_nodata, dtype=np.float32)
    
    has_valid_pixels = np.any(valid)
    
    if has_valid_pixels:
        ndwi[valid] = (b2[valid] - b3[valid]) / denominator[valid]

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        actual_path, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if out_ds is None:
        raise RuntimeError(f"Cannot create NDWI raster: {actual_path}")

    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(ndwi)
    out_band.SetNoDataValue(float(ndwi_nodata))
    
    # Вычисляем статистику только если есть валидные пиксели
    if has_valid_pixels:
        try:
            out_band.ComputeStatistics(False)
        except:
            pass  # Игнорируем ошибки статистики
    
    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    ds = None

    if has_valid_pixels:
        print(f"  ✓ Растр NDWI сохранён: {os.path.basename(actual_path)}")
    else:
        print(f"  ⚠ Растр NDWI сохранён (без валидных пикселей): {os.path.basename(actual_path)}")
    
    return actual_path


def _add_raster_to_qgis_map(raster_path, layer_name):
    """Добавляет растровый слой в проект QGIS."""
    if not HAS_QGIS:
        return

    layer = QgsRasterLayer(raster_path, layer_name)
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        print(f"  ✓ Слой добавлен в QGIS: {layer_name}")
    else:
        print(f"  ⚠ Не удалось добавить слой в QGIS: {layer_name}")


def extract_year_from_path(filepath):
    """Извлекает год из пути."""
    match = re.search(r'Петрунь_(\d{4})', filepath)
    if match:
        return match.group(1)
    basename = os.path.basename(filepath)
    match = re.search(r'(\d{4})', basename)
    return match.group(1) if match else "unknown"


def get_base_name(filepath):
    """Получает базовое имя файла без расширения."""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_vegetation_raster_path(year):
    """Возвращает путь к растру растительности для указанного года."""
    return rf"C:\Users\User\Downloads\IKI_rastitelnost\ИКИ_растительность\spbu_{year}.tif"


def process_single_image(
    input_raster_path,
    output_dirs,
    buffer_gpkg_path,
    cloud_params=None,
    add_to_qgis=True,
):
    """
    Обрабатывает один снимок Метеор-М с обрезкой по буферу 20 км.
    Все промежуточные маски создаются в памяти.
    На диск сохраняются только: обрезанный снимок и растр NDWI.
    """
    if cloud_params is None:
        cloud_params = {}
    
    base_name = get_base_name(input_raster_path)
    year = extract_year_from_path(input_raster_path)
    region = "Петрунь"
    date_prefix = f"{region} {year} - {base_name}"
    
    print(f"\n{'='*60}")
    print(f"Обработка: {base_name}")
    print(f"Регион: {region}, Год: {year}")
    print(f"Буфер: 20 км вокруг метеостанции")
    print(f"{'='*60}")
    
    try:
        if not os.path.exists(input_raster_path):
            raise FileNotFoundError(f"Файл не найден: {input_raster_path}")
        
        if not os.path.exists(buffer_gpkg_path):
            raise FileNotFoundError(f"Файл буфера не найден: {buffer_gpkg_path}")
        
        vegetation_raster = get_vegetation_raster_path(year)
        if not os.path.exists(vegetation_raster):
            raise FileNotFoundError(f"Растр растительности не найден: {vegetation_raster}")
        
        print(f"  → Открытие снимка...")
        reference_ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
        if reference_ds is None:
            raise RuntimeError(f"Не удалось открыть снимок: {input_raster_path}")
        
        # 0. Растеризация буфера
        print(f"  → Растеризация буфера 20 км...")
        buffer_mask = _rasterize_vector_to_mask(buffer_gpkg_path, reference_ds)
        buffer_pixels = np.sum(buffer_mask)
        print(f"  ✓ Буфер растеризован: {buffer_pixels} пикселей")
        
        # 1. Маска облаков В ПАМЯТИ
        print(f"  → Создание маски облаков (в памяти)...")
        
        if reference_ds.RasterCount < 6:
            raise RuntimeError(f"Expected at least 6 bands, found: {reference_ds.RasterCount}")
        
        cols = reference_ds.RasterXSize
        rows = reference_ds.RasterYSize
        
        b1 = reference_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        b2 = reference_ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        b3 = reference_ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
        b4 = reference_ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
        b5 = reference_ds.GetRasterBand(5).ReadAsArray().astype(np.float32)
        b6 = reference_ds.GetRasterBand(6).ReadAsArray().astype(np.float32)
        
        nodata_mask = np.zeros((rows, cols), dtype=bool)
        for arr in (b1, b2, b3, b4, b5, b6):
            nodata_mask |= ~np.isfinite(arr)
        
        cloud_mask = _night_cloud_mask_northern(b1, b2, b3, b4, b5, b6, **cloud_params)
        cloud_mask = np.where(nodata_mask, False, cloud_mask.astype(bool))
        
        print(f"  ✓ Маска облаков создана в памяти")
        
        # 2. Маска растительности В ПАМЯТИ
        print(f"  → Выравнивание маски растительности...")
        shrub_mask = _read_shrub_mask_aligned(vegetation_raster, reference_ds)
        print(f"  ✓ Маска растительности загружена в память")
        
        # 3. Кустарники без облаков В ПАМЯТИ
        shrub_no_cloud_mask = shrub_mask & (~cloud_mask)
        shrub_pixels = np.sum(shrub_no_cloud_mask)
        print(f"  ✓ Кустарников без облаков: {shrub_pixels} пикселей")
        
        # 4. Финальная маска (буфер + кустарники без облаков)
        final_mask = shrub_no_cloud_mask & buffer_mask
        final_pixels = np.sum(final_mask)
        print(f"  ✓ Пикселей в финальной маске (буфер 20км + кустарники без облаков): {final_pixels}")
        
        # 5. Обрезанный снимок (только если есть пиксели)
        if final_pixels > 0:
            cropped_path = os.path.join(
                output_dirs['cropped'],
                f"Обрезанный снимок - {date_prefix}.tif"
            )
            print(f"  → Создание обрезанного снимка...")
            _write_masked_raster(input_raster_path, cropped_path, final_mask)
            if add_to_qgis:
                _add_raster_to_qgis_map(cropped_path, f"Снимок обрезанный {region} {year} - {base_name}")
        else:
            print(f"  ⚠ Нет пикселей в буфере, обрезанный снимок не создаётся")
        
        # 6. NDWI растр
        ndwi_path = os.path.join(
            output_dirs['ndwi'],
            f"NDWI - {date_prefix}.tif"
        )
        print(f"  → Расчёт NDWI в буфере 20 км...")
        ndwi_path = _write_ndwi_raster_in_buffer(
            input_raster_path, ndwi_path, 
            buffer_mask=buffer_mask, 
            shrub_mask=shrub_no_cloud_mask
        )
        if add_to_qgis:
            _add_raster_to_qgis_map(ndwi_path, f"NDWI {region} {year} - {base_name}")
        
        # 7. Средний NDWI
        mean_ndwi, pixel_count = _calculate_ndwi_mean_in_buffer(
            input_raster_path, 
            buffer_mask=buffer_mask,
            shrub_mask=shrub_no_cloud_mask
        )
        
        reference_ds = None
        
        if mean_ndwi is not None:
            print(f"  ✓ Средний NDWI (в буфере 20 км): {mean_ndwi:.6f}")
            print(f"  ✓ Количество пикселей: {pixel_count}")
        else:
            print(f"  ⚠ Нет валидных пикселей для расчёта NDWI в буфере")
        
        return base_name, mean_ndwi, pixel_count, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ ОШИБКА: {error_msg}")
        return base_name, None, 0, error_msg


def main():
    """Главная функция пакетной обработки снимков Метеор-М"""
    
    # ===== НАСТРОЙКИ =====
    
    input_rasters = [
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_03320_3315_230220.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_4422_110520.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_6420_280920.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_3475_050320.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_4627_250520.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_3845_310320.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_5418_200720.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_4030_130420.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_5438_210720.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_4044_140420.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_5674_070820.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_4058_150420.tiff",
        r"C:\Users\User\Downloads\METEOR\METEOR\Петрунь_2020\METM22_6391_260920.tiff",
    ]
    
    buffer_gpkg = r"C:\Users\User\Downloads\METEOR\METEOR\Буфериз.gpkg"
    
    output_dirs = {
        'cropped': r"C:\Users\User\Downloads\METEOR\Обрезанный снимок (только по маске)",
        'ndwi': r"C:\Users\User\Downloads\METEOR\Растр индекса NDWI (результат)",
    }
    
    cloud_params = {
        'cold_k': 0.75,
        'texture_k': 0.5,
        'ratio_k': 0.6,
        'min_votes': 4,
        'dark_min_votes': 3,
        'dark_seed_votes': 4,
        'bright_min_votes': 3,
        'bright_seed_votes': 4,
        'min_neighbors': 0,
    }
    
    # ===== ПРОВЕРКА ДИРЕКТОРИЙ =====
    
    print(f"\n{'#'*60}")
    print(f"Проверка выходных директорий...")
    print(f"{'#'*60}")
    
    for dir_name, dir_path in output_dirs.items():
        if not os.path.exists(dir_path):
            print(f"  ✗ НЕ НАЙДЕНА: {dir_name} — {dir_path}")
            print(f"  Создайте папку и запустите скрипт снова.")
            return
        else:
            print(f"  ✓ {dir_name}")
    
    # ===== ПРОВЕРКА ВХОДНЫХ ДАННЫХ =====
    
    print(f"\n{'#'*60}")
    print(f"Проверка входных данных...")
    print(f"{'#'*60}")
    
    if not os.path.exists(buffer_gpkg):
        print(f"  ✗ Файл буфера не найден: {buffer_gpkg}")
        return
    else:
        print(f"  ✓ Буфер 20 км: {buffer_gpkg}")
    
    existing_rasters = []
    
    for raster_path in input_rasters:
        if not os.path.exists(raster_path):
            print(f"  ⚠ Не найден: {raster_path}")
            continue
        
        year = extract_year_from_path(raster_path)
        veg_path = get_vegetation_raster_path(year)
        
        if not os.path.exists(veg_path):
            print(f"  ⚠ Нет растра растительности для {os.path.basename(raster_path)} (год {year})")
            continue
        
        existing_rasters.append(raster_path)
    
    print(f"\n  ✓ Доступно для обработки: {len(existing_rasters)} из {len(input_rasters)} снимков")
    
    if not existing_rasters:
        print(f"\n  ✗ Нет доступных снимков для обработки!")
        return
    
    # ===== ПАКЕТНАЯ ОБРАБОТКА =====
    
    print(f"\n{'#'*60}")
    print(f"ЗАПУСК ПАКЕТНОЙ ОБРАБОТКИ")
    print(f"Всего снимков: {len(existing_rasters)}")
    print(f"Буфер: 20 км вокруг метеостанции Петрунь")
    print(f"Сохраняются только: обрезанный снимок и NDWI")
    print(f"{'#'*60}")
    
    results = []
    
    for idx, raster_path in enumerate(existing_rasters, 1):
        print(f"\n[{idx}/{len(existing_rasters)}] {os.path.basename(raster_path)}")
        
        base_name, mean_ndwi, pixel_count, error = process_single_image(
            input_raster_path=raster_path,
            output_dirs=output_dirs,
            buffer_gpkg_path=buffer_gpkg,
            cloud_params=cloud_params,
            add_to_qgis=HAS_QGIS,
        )
        
        results.append({
            'name': base_name,
            'ndwi': mean_ndwi,
            'pixels': pixel_count,
            'error': error,
            'path': raster_path,
        })
    
    # ===== ВЫВОД РЕЗУЛЬТАТОВ =====
    
    print(f"\n{'#'*60}")
    print(f"РЕЗУЛЬТАТЫ ОБРАБОТКИ (буфер 20 км, Петрунь)")
    print(f"{'#'*60}")
    
    print(f"\n{'Имя снимка':<35} {'NDWI':>10} {'Пикселей':>10} {'Статус':>15}")
    print("-" * 72)
    
    for result in results:
        name = result['name'][:34]
        if result['error']:
            print(f"{name:<35} {'Н/Д':>10} {'-':>10} {'ОШИБКА':>15}")
        elif result['ndwi'] is not None:
            print(f"{name:<35} {result['ndwi']:>10.6f} {result['pixels']:>10} {'Успешно':>15}")
        else:
            print(f"{name:<35} {'Н/Д':>10} {result['pixels']:>10} {'Нет данных':>15}")
    
    # Сохраняем CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dirs['ndwi'], f"NDWI_Петрунь_буфер20км_{timestamp}.csv")
    csv_path = _get_safe_output_path(csv_path)
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Имя снимка', 'Путь', 'Средний NDWI (буфер 20км)', 'Пикселей в буфере', 'Статус', 'Ошибка'])
        for result in results:
            writer.writerow([
                result['name'],
                result['path'],
                result['ndwi'] if result['ndwi'] is not None else '',
                result['pixels'],
                'Успешно' if result['error'] is None else 'Ошибка',
                result['error'] if result['error'] else '',
            ])
    
    print(f"\n✓ CSV сохранён: {os.path.basename(csv_path)}")
    
    # Статистика
    valid_results = [r for r in results if r['ndwi'] is not None and r['error'] is None]
    
    if valid_results:
        ndwi_values = [r['ndwi'] for r in valid_results]
        pixel_counts = [r['pixels'] for r in valid_results]
        
        print(f"\n{'='*60}")
        print(f"СТАТИСТИКА NDWI (буфер 20 км, Петрунь)")
        print(f"{'='*60}")
        print(f"  Снимков с данными: {len(valid_results)} из {len(results)}")
        print(f"  Средний NDWI:       {np.mean(ndwi_values):.6f}")
        print(f"  Мин NDWI:           {np.min(ndwi_values):.6f}")
        print(f"  Макс NDWI:          {np.max(ndwi_values):.6f}")
        print(f"  Станд. откл:        {np.std(ndwi_values):.6f}")
        print(f"  Медиана:            {np.median(ndwi_values):.6f}")
        print(f"  Среднее пикселей:   {np.mean(pixel_counts):.0f}")
    
    error_results = [r for r in results if r['error'] is not None]
    if error_results:
        print(f"\n  ⚠ Снимков с ошибками: {len(error_results)}")
        for r in error_results:
            print(f"    - {r['name']}: {r['error']}")
    
    print(f"\n{'='*60}")
    print(f"ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"{'='*60}")


# Запуск
print("\n" + "="*60)
print("ОБРАБОТКА СНИМКОВ МЕТЕОР-М")
print("Буфер 20 км вокруг метеостанции Петрунь")
print("Классы растительности: 11, 12, 14")
print("="*60 + "\n")
main()
