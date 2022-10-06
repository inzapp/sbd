from glob import glob


def main():
    arr = []
    for path in glob('*.h5'):
        name = path[:-3]
        if name.find('mAP') > -1:
            sp = name.split('_')
            mAP = float(sp[4])
            f1 = float(sp[6])
            iou = float(sp[9])
            tp = int(sp[11])
            fp = int(sp[13])
            arr.append({
                'rank' : 0,
                'name' : path,
                'mAP' : mAP,
                'f1' : f1,
                'iou' : iou,
                'tp' : tp,
                'fp' : fp})

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

