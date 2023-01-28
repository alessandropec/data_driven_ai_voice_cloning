from numpy import array as np_array
from numpy import float as np_float
colormap = np_array([
    [255, 0, 0], # rosso
    [255, 128, 0], # arancione
    [255, 255, 0], # giallo
    [128, 255, 0], # lime
    [0, 255, 0], # verde chiaro
    [0, 255, 128], # verde acqua
    [0, 255, 255], # ciano
    [0, 128, 255], # azzurro
    [0, 0, 255], # blu
    [128, 0, 255], # indigo
    [255, 0, 255], # viola
    [255, 0, 128], # rosa
    [240, 128, 128], # rosa scuro
    [255, 99, 71], # corallo
    [255, 69, 0], # arancione scuro
    [255, 215, 0], # oro
    [184, 134, 11], # marrone
    [139, 69, 19], # marrone scuro
    [128, 0, 0], # rosso scuro
    [128, 0, 128], # viola scuro
    [85, 107, 47], # verde oliva scuro
    [47, 79, 79], # grigio scuro
    [0, 0, 0], # nero
    [105, 105, 105], # grigio
    [112, 128, 144], # grigio chiaro
    [119, 136, 153], # grigio medio
    [190, 190, 190], # grigio chiaro
    [245, 245, 220], # beige
    [222, 184, 135], # marrone chiaro
    [210, 180, 140], # marrone medio
    [188, 143, 143], # rosa pallido
    [218, 165, 32], # oro chiaro
    [205, 133, 63], # arancione medio
    [139, 69, 19], # marrone scuro
    [160, 82, 45], # marrone scuro
    [165, 42, 42], # rosso scuro
    [178, 34, 34], # rosso scuro
    [220, 20, 60], # rosso scuro
    [255, 0, 0], # rosso
    [255, 105, 180], # rosa
    [255, 20, 147], # rosa scuro
    [255, 192, 203], # rosa pallido
    [255, 182, 193], # rosa pallido
    [255, 250, 250], # rosa pallido
    [245, 255, 250], # rosa pallido
    [240, 255, 240], # verde chiaro
    [240, 248, 255], # azzurro chiaro
    [240, 230, 140], # kaki
    [230, 230, 250], # azzurro chiaro
    [224, 255, 255], # ciano chiaro
    [216, 191, 216], # viola chiaro
    [211, 211, 211], # grigio medio
    [192, 192, 192], # grigio medio
    [188, 143, 143], # rosa pallido
    [186, 85, 211], # viola scuro
    [135, 206, 235], # azzurro
    [135, 206, 250], # azzurro chiaro
    [0, 100, 0], # verde scuro
    [34, 139, 34], # verde scuro
    [0, 128, 0], # verde medio
    [0, 255, 0], # verde chiaro
    [107, 142, 35], # verde oliva scuro
    [124, 252, 0], # lime verde
    [127, 255, 0], # lime
    [173, 255, 47], # verde chiaro
    [50, 205, 50], # verde medio
    [144, 238, 144], # verde chiaro
    [152, 251, 152], # verde chiaro
    [143, 188, 143], # verde chiaro
    [0, 250, 154], # verde medio
    [0, 255, 127], # verde acqua
    [0, 201, 87], # verde acqua scuro
    [46, 139, 87], # verde scuro
    [60, 179, 113], # verde medio
    [32, 178, 170], # verde acqua
    [47, 79, 79], # grigio scuro
    [0, 128, 128], # verde acqua scuro
    [0, 139, 139], # verde acqua scuro
    [0, 255, 255], # ciano
    [0, 255, 255], # ciano
    [224, 255, 255], # ciano chiaro
    [95, 158, 160], # ciano scuro
    [100, 149, 237], # blu scuro
    [0, 0, 205], # blu medio
    [0, 0, 139], # blu scuro
    [0, 0, 128], # blu scuro
    [25, 25, 112], # blu scuro
    [65, 105, 225], # blu scuro
    [138, 43, 226], # blu viola
    [75, 0, 130], # indigo
    [72, 61, 139], # blu scuro
    [106, 90, 205], # blu scuro
    [123, 104, 238], # blu scuro
    [147, 112, 219], # viola scuro
    [139, 0, 139], # viola scuro
    [148, 0, 211], # viola scuro
    [153, 50, 204], # viola scuro
    [186, 85, 211], # viola scuro
    [128, 0, 128], # viola scuro
    [216, 191, 216], # viola chiaro
    [221, 160, 221], # viola scuro
    [238, 130, 238], # viola
    [255, 0, 255], # viola
    [218, 112, 214], # viola scuro
    [255, 0, 255], # viola
    [199, 21, 133], # viola scuro
    [219, 112, 147], # rosa scuro
    [255, 20, 147], # rosa scuro
    [255, 105, 180], # rosa
    [255, 192, 203], # rosa pallido
    [255, 182, 193], # rosa pallido
    [255, 250, 250], # rosa pallido
    [245, 255, 250], # rosa pallido
    [240, 255, 240], # verde chiaro
    [240, 248, 255], # azzurro chiaro
    [240, 230, 140], # kaki
    [230, 230, 250], # azzurro chiaro
        
], dtype=np_float) / 255