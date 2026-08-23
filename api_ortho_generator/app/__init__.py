"""
OrthoGenerator 2D Orthophoto Microservice
"""
import os

try:
    import rasterio
    proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    gdal_dir = os.path.join(os.path.dirname(rasterio.__file__), "gdal_data")
    if os.path.exists(proj_dir):
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"] = proj_dir
        try:
            from rasterio._env import set_gdal_config
            set_gdal_config("PROJ_LIB", proj_dir)
            set_gdal_config("PROJ_DATA", proj_dir)
        except Exception:
            pass
    if os.path.exists(gdal_dir):
        os.environ["GDAL_DATA"] = gdal_dir
except Exception:
    pass

__version__ = "1.0.0"
