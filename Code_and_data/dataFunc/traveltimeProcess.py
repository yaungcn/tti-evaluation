import numpy as np


def classify_ttdata_by_level(inv_ttdata_all, ref_ttdata_all, n_all=200):
    inv_ttdata_all = np.array(inv_ttdata_all)
    ref_ttdata_all = np.array(ref_ttdata_all)

    row, col = 5, 10

    inv = {k: [] for k in [
        'l0', 'l0_0', 'l0_1', 'l0_2',
        'l1', 'l1_0', 'l1_1', 'l1_2', 'l1_3',
        'l2', 'l2_0', 'l2_1', 'l2_2',
        'l3', 'l3_0', 'l3_1',
        'l4',
    ]}
    ref = {k: [] for k in inv}

    for i in range(n_all):
        for r in range(row):
            for c in range(col):
                mod = int(np.abs(r - np.mod(c, row)))
                if mod == 0:
                    inv['l0'].append(inv_ttdata_all[i][r][c])
                    ref['l0'].append(ref_ttdata_all[i][r][c])
                    if r == 0 or r == 4:
                        inv['l0_0'].append(inv_ttdata_all[i][r][c])
                        ref['l0_0'].append(ref_ttdata_all[i][r][c])
                    if r == 1 or r == 3:
                        inv['l0_1'].append(inv_ttdata_all[i][r][c])
                        ref['l0_1'].append(ref_ttdata_all[i][r][c])
                    if r == 2:
                        inv['l0_2'].append(inv_ttdata_all[i][r][c])
                        ref['l0_2'].append(ref_ttdata_all[i][r][c])
                elif mod == 1:
                    inv['l1'].append(inv_ttdata_all[i][r][c])
                    ref['l1'].append(ref_ttdata_all[i][r][c])
                    if r == 0 or r == 4:
                        inv['l1_0'].append(inv_ttdata_all[i][r][c])
                        ref['l1_0'].append(ref_ttdata_all[i][r][c])
                    if r == 1 or r == 3:
                        if c == 0 or c == 5 or c == 4 or c == 9:
                            inv['l1_1'].append(inv_ttdata_all[i][r][c])
                            ref['l1_1'].append(ref_ttdata_all[i][r][c])
                        if c == 2 or c == 7:
                            inv['l1_2'].append(inv_ttdata_all[i][r][c])
                            ref['l1_2'].append(ref_ttdata_all[i][r][c])
                    if r == 2:
                        inv['l1_3'].append(inv_ttdata_all[i][r][c])
                        ref['l1_3'].append(ref_ttdata_all[i][r][c])
                elif mod == 2:
                    inv['l2'].append(inv_ttdata_all[i][r][c])
                    ref['l2'].append(ref_ttdata_all[i][r][c])
                    if r == 0 or r == 4:
                        if c == 2 or c == 7:
                            inv['l2_0'].append(inv_ttdata_all[i][r][c])
                            ref['l2_0'].append(ref_ttdata_all[i][r][c])
                    if r == 1 or r == 3:
                        if c == 3 or c == 8:
                            inv['l2_1'].append(inv_ttdata_all[i][r][c])
                            ref['l2_1'].append(ref_ttdata_all[i][r][c])
                    if r == 2:
                        if c == 0 or c == 5 or c == 4 or c == 9:
                            inv['l2_2'].append(inv_ttdata_all[i][r][c])
                            ref['l2_2'].append(ref_ttdata_all[i][r][c])
                elif mod == 3:
                    inv['l3'].append(inv_ttdata_all[i][r][c])
                    ref['l3'].append(ref_ttdata_all[i][r][c])
                    if c == 3 or c == 8 or c == 1 or c == 6:
                        inv['l3_0'].append(inv_ttdata_all[i][r][c])
                        ref['l3_0'].append(ref_ttdata_all[i][r][c])
                    if c == 0 or c == 5 or c == 4 or c == 9:
                        inv['l3_1'].append(inv_ttdata_all[i][r][c])
                        ref['l3_1'].append(ref_ttdata_all[i][r][c])
                elif mod == 4:
                    inv['l4'].append(inv_ttdata_all[i][r][c])
                    ref['l4'].append(ref_ttdata_all[i][r][c])

    return {"inv": inv, "ref": ref}
