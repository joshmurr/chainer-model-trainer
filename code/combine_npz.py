#!/usr/bin/env python3
import numpy as np
import os
import tqdm

def getBatchSize(source, dryRun):
  tmp = []
  with np.load(source) as data:
    for key in data:
      if dryRun:
        print(f"SHAPE: {data[key].shape}") # (2025, 3, 128, 128)
        print(f"dtype: {data[key].dtype}") # uint8
      tmp.append(data[key])

  size = [x.size * x.itemsize for x in tmp]
  return size



def process(output, sources, lim, dryRun, force=False):
    data = []

    if not force:
      if os.path.isfile(output):
        msg = 'error opening target file (does {} exist?).\n'.format(output)
        msg += 'Pass "-f" argument to overwrite output file.'
        raise ValueError(msg)

    sources = sources[:lim]

    first_batch_size = getBatchSize(sources[0], dryRun)[0]
    print(f"Size of one batch: {first_batch_size}")
    print(f"Size of total: {(first_batch_size * len(sources)) * 1e-9}GB")

    if dryRun:
      return

    k = None
    # Loop over the source files
    for source in tqdm.tqdm(sources):
      # print('Source file {}: {}'.format(i + 1, source))
      with np.load(source) as loaded:
        # Keep the arrays
        for key in loaded.files:
          k = key
          if loaded[key].ndim == 4:
            data.append(loaded[key])

    data = np.concatenate(data, axis=0)
    np.savez_compressed(output, **{'%s' % k: data})

    print('done')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='For combining NPZ files.')
  parser.add_argument('-f', '--force', action='store_true', help='force write the target file')
  parser.add_argument('-o', '--output', help='output filename')
  parser.add_argument('-s', '--sources', nargs='+', help='source files')
  parser.add_argument('-l', '--lim', type=int, default=-1, help='max number of batches to process')
  parser.add_argument('-d', '--dryRun', type=bool, default=False, help='dont do the thing')
  args = parser.parse_args()

  process(args.output, args.sources, args.lim, args.dryRun, force=args.force)
