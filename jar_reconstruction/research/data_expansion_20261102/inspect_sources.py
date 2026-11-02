"""只读ZIP目录，不下载整个训练包；输出选取及空间规划所需目录。"""
import json
import zipfile
from direct_http import ROOT, RemoteZipReader

sources = [
    ('dynamic_train001', 'https://hf-mirror.com/datasets/geyongtao/dynamic_replica/resolve/main/train_001.zip',
     next(x['size'] for x in json.loads((ROOT / 'mvs_data_dynamic_geyongtao.json').read_text()) if x['path'] == 'train_001.zip')),
    ('hypersim_probe', 'https://modelscope.cn/datasets/OpenDataLab/Hypersim/resolve/master/raw/ai_010_006.zip', 1141950961),
]
for name, url, size in sources:
    with zipfile.ZipFile(RemoteZipReader(url, size)) as archive:
        info = [dict(name=x.filename, bytes=x.file_size, compressed=x.compress_size,
                     offset=x.header_offset, crc=x.CRC) for x in archive.infolist()]
        (ROOT / (name + '_members.json')).write_text(json.dumps(info))
        print(name, len(info), 'members', sum(x['bytes'] for x in info), 'bytes', flush=True)
        print([x['name'] for x in info[:15]], flush=True)
