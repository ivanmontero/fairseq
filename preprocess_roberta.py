import torch
import os
import subprocess

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--lang", type=str)
parser.add_argument("--trainpref", type=str)
parser.add_argument("--validpref", type=str)
parser.add_argument("--testpref", type=str)
parser.add_argument("--destdir", type=str)
parser.add_argument("--workers", type=int)
args = parser.parse_args()

subprocess.call(f"mkdir -p {args.destdir}", shell=True)
subprocess.call(f"wget -O {os.path.join(args.destdir,'encoder.json')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json", shell=True)
subprocess.call(f"wget -O {os.path.join(args.destdir,'vocab.bpe')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe", shell=True)
subprocess.call(f"wget -O {os.path.join(args.destdir,'roberta.dict.txt')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt", shell=True)

print("Encoding train file")
subprocess.call(
    f"""python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json {os.path.join(args.destdir,'encoder.json')} \
        --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
        --inputs {args.trainpref + "." + args.lang} \
        --outputs {os.path.join(args.destdir, 'train.bpe.' + args.lang)}  \
        --keep-empty \
        --workers {args.workers}""", shell=True)
data_commands = f"--trainpref {os.path.join(args.destdir, 'train.bpe')}"

if args.validpref is not None:
    print("Encoding valid file")
    subprocess.call(
        f"""python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json {os.path.join(args.destdir,'encoder.json')} \
            --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
            --inputs {args.validpref + "." + args.lang} \
            --outputs {os.path.join(args.destdir, 'valid.bpe.' + args.lang)}  \
            --keep-empty \
            --workers {args.workers}""", shell=True)
    data_commands += f" --validpref {os.path.join(args.destdir, 'valid.bpe')}"

if args.testpref is not None:
    print("Encoding test file")
    subprocess.call(
        f"""python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json {os.path.join(args.destdir,'encoder.json')} \
            --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
            --inputs {args.testpref + "." + args.lang} \
            --outputs {os.path.join(args.destdir, 'test.bpe.' + args.lang)}  \
            --keep-empty \
            --workers {args.workers}""", shell=True)
    data_commands += f" --testpref {os.path.join(args.destdir, 'test.bpe')}"

print("Executing fairseq preprocessing")
subprocess.call(
    f"""fairseq-preprocess \
        --source-lang {args.lang} --target-lang {args.lang} \
        {data_commands} \
        --destdir {args.destdir} \
        --workers {args.workers} \
        --srcdict {os.path.join(args.destdir, "roberta.dict.txt")} \
        --tgtdict {os.path.join(args.destdir, "roberta.dict.txt")}""", shell=True)


# print("Downloading roberta")
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
# print("Extracting roberta vocab and bpe")
# roberta.task.dictionary.save(os.path.join(args.destdir, "roberta.dict.txt"))
# bpe = roberta.bpe

# print("Encoding train file")
# os.makedirs(args.destdir, exist_ok=True)
# with open(args.trainpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "train.bpe." + args.lang), "w+") as tf:
#     for l in sf:
#         tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
# data_commands = f"--trainpref {os.path.join(args.destdir, 'train.bpe')}"

# if args.validpref is not None:
#     print("Encoding valid file")
#     with open(args.validpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "valid.bpe." + args.lang), "w+") as tf:
#         for l in sf:
#             tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
#     data_commands += f" --validpref {os.path.join(args.destdir, 'valid.bpe')}"

# if args.testpref is not None:
#     print("Encoding test file")
#     with open(args.testpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "test.bpe." + args.lang), "w+") as tf:
#         for l in sf:
#             tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
#     data_commands += f" --testpref {os.path.join(args.destdir, 'test.bpe')}"

# print("Executing fairseq preprocessing")
# subprocess.call(
# f"""fairseq-preprocess \
#         --source-lang {args.lang} --target-lang {args.lang} \
#         {data_commands} \
#         --destdir {args.destdir} \
#         --workers {args.workers} \
#         --srcdict {os.path.join(args.destdir, "roberta.dict.txt")} --tgtdict {os.path.join(args.destdir, "roberta.dict.txt")}
# """, shell=True)