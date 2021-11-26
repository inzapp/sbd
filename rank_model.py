from glob import glob


def sort_by_key_and_add_score(arr, key, reverse):
    arr = sorted(arr, key=lambda x: x[key], reverse=reverse)
    for i in range(len(arr)):
        arr[i]['rank_sum'] += i


def main():
    arr = []
    for path in glob('*.h5'):
        name = path[:-3]
        sp = name.split('_')
        mAP = float(sp[4])
        f1 = float(sp[6])
        iou = float(sp[9])
        tp = int(sp[11])
        fp = int(sp[13])
        arr.append({
            'rank_sum' : 0,
            'name' : path,
            'mAP' : mAP,
            'f1' : f1,
            'iou' : iou,
            'tp' : tp,
            'fp' : fp})

    # sort_by_key_and_add_score(arr, 'mAP', reverse=True)
    # sort_by_key_and_add_score(arr, 'f1', reverse=True)
    # sort_by_key_and_add_score(arr, 'iou', reverse=True)
    sort_by_key_and_add_score(arr, 'tp', reverse=True)
    sort_by_key_and_add_score(arr, 'fp', reverse=False)

    arr = sorted(arr, key=lambda x: x['rank_sum'], reverse=True)
    for node in arr:
        print(f'rank_sum : {node["rank_sum"]}, name : {node["name"]}')
    print('\nlower rank_sum is better')

if __name__ == '__main__':
    main()
