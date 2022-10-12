import os
from glob import glob


def parse(basename, sp, name):
    return float(sp[sp.index(name)+1]) if basename.find(name) > -1 else 0.0


def main():
    arr = []
    for path in glob('*.h5'):
        basename = os.path.basename(path)
        sp = basename.split('_')
        mAP  = parse(basename, sp, 'mAP')
        f1   = parse(basename, sp, 'f1')
        iou  = parse(basename, sp, 'iou')
        tp   = parse(basename, sp, 'tp')
        fp   = parse(basename, sp, 'fp')
        conf = parse(basename, sp, 'conf')
        arr.append({
            'rank': 0,
            'name': path,
            'mAP': mAP,
            'f1': f1,
            'iou': iou,
            'tp': tp,
            'fp': fp,
            'conf' : conf})

    arr = sorted(arr, key=lambda x: x['conf'], reverse=True)
    arr = sorted(arr, key=lambda x: x['iou'], reverse=True)
    arr = sorted(arr, key=lambda x: x['mAP'], reverse=True)
    arr = sorted(arr, key=lambda x: x['f1'], reverse=True)
    arr = sorted(arr, key=lambda x: x['fp'], reverse=False)
    arr = sorted(arr, key=lambda x: x['tp'], reverse=True)
    for i in range(len(arr)):
        arr[i]['rank'] += i + 1

    arr = sorted(arr, key=lambda x: x['rank'], reverse=True)
    for node in arr:
        print(f'rank : {node["rank"]}, name : {node["name"]}')
    print('\nlower rank is better')


if __name__ == '__main__':
    main()

