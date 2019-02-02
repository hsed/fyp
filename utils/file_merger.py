import fileinput

## note default paths are w.r.t main directory so its assume script call is as `python lib\file_merger.py`

num_sub = 9
first_fname = 'eval/MSRA15/eval_test_0_ahpe.txt'
out_fname = 'eval/MSRA15/eval_all_ahpe_deep_prior_pp_xyz.txt'

filenames = [first_fname.replace('0', str(i)) for i in range(1,9)]

print(filenames)

with open(out_fname, 'w') as fout, fileinput.input(filenames) as fin:
    for line in fin:
        fout.write(line)