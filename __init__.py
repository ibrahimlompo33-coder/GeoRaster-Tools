def classFactory(iface):
    from .raster_vectorizer import RasterVectorizerPlugin
    return RasterVectorizerPlugin(iface)
