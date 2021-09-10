import numpy as np
from tqdm import tqdm

for split in ['valid', 'test', 'train']:
    num_list = [int(line.strip()) for line in open('data_step0/' + split + '.seg_num').readlines()]
    src_list = [line.strip() for line in open('bpe_src_kb/' + split + '.source_kb_val').readlines()]
    tgt_list = [line.strip() for line in open('bpe_src_kb/' + split + '.target').readlines()]
    sku_list = [line.strip() for line in open('data_step0/' + split + '.skus').readlines()]
    cursor = 0
    src_arr = []
    tgt_arr = []
    type_arr = []
    img_arr = []
    for num in tqdm(num_list, desc=split):
        assert num > 1
        skus = sku_list[cursor: cursor + num]
        #src_txt = ' [SEP] '.join(src_list[cursor: cursor + num]).strip() + ' [SEP]'
        src_txt = []
        src_txt += src_list[cursor].split()[:199]
        if len(src_txt) < 199:
            src_txt += ['[PAD]' for _ in range(199 - len(src_list[cursor].split()))]
        src_txt.append('[SEP]')

        assert len(src_txt) == 200

        src_txt += src_list[cursor + 1].split()[:99]
        if len(src_txt) < 299:
            src_txt += ['[PAD]' for _ in range(99 - len(src_list[cursor + 1].split()))]
        src_txt.append('[SEP]')

        assert len(src_txt) == 300

        src_remain = ' '.join(src_list[cursor + 2: cursor + num]).split()
        src_txt += src_remain[:99]
        if len(src_txt) < 399:
            src_txt += ['[PAD]' for _ in range(99 - len(src_remain))]
        src_txt.append('[SEP]')
        
        assert len(src_txt) == 400
        
        src_txt = ' '.join(src_txt).strip()
        
        tgt_txt = tgt_list[cursor]
        img_txt = ''
        for sku in skus:
            if img_txt:
                img_txt += ' '
            #img_txt += str(image_name_idx[sku])
            img_txt += sku
        type_txt = ''
        counter = -1
        for src in src_list[cursor: cursor + num]:
            if type_txt:
                type_txt += ' '
            counter += 1
            type_txt += ' '.join([str(counter) for _ in range(len(src.strip().split()))])
        cursor += num
        #src_arr.append(src_txt.strip() + ' ' + ' '.join(['[PAD]' for _ in range(num)]) + '\n')
        src_arr.append(src_txt.strip() + '\n')
        tgt_arr.append(tgt_txt.strip() + '\n')
        #type_arr.append(type_txt.strip() + ' ' + ' '.join([str(counter) for _ in range(num)]) + '\n')
        type_arr.append(type_txt.strip() + '\n')
        img_arr.append(img_txt.strip() + '\n')

        # assert len(type_txt.split()) == len(src_txt.split())   # 不使用type_embedding, type 少了 source 中新增的 [SEP] 
    with open('data_step1/' + split + '.source', 'w') as f:
        f.writelines(src_arr)
    with open('data_step1/' + split + '.target', 'w') as f:
        f.writelines(tgt_arr)
    with open('data_step1/' + split + '.sku2vec', 'w') as f:
        f.writelines(type_arr)
    with open('data_step1/' + split + '.img2ids', 'w') as f:
        f.writelines(img_arr)
