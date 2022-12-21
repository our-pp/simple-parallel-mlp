import os, time
import matplotlib.pyplot as plt

batch_size = [
    10,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    4096 + 2048,
    8192,
    8192 + 2048,
    8192 + 4096,
    8192 + 4096 + 2048,
    16384,
]
# batch_size = [512, 512+256, 512+256+128, 1024, 1024+256, 1024+512, 2048, 4096]
# batch_size = [4096, 4096+512, 4096+1024, 4096+1024+512, 4096+2048, 4096+2048+512, 4096+2048+1024, 4096+2048+1024+512, 6144]
batch_size = range(16, 17)
hidden_dim = [100, 200, 300, 400, 500, 600]
train_size = [60000]


# def draw_line_chart(y, x) -> None:
#     plt.plot(x, y)
#     plt.title(())
#     plt.xlabel("Epoch")output_filename.removesuffix(".png").replace("-", " ").capitalize
#     plt.ylabel("Loss")
#     plt.savefig(IMAGE_DIR / output_filename)


def iter_batch(method) -> None:
    for train in train_size:
        print(f"train size: {train}")
        for batch in batch_size:
            # for batch in range(1000, 40000+1, 1000):
            sec = os.popen(f"echo -1 | ./bin/{method} -b {batch} -t {train}").read()
            print(
                f"    one epoch for batch size: {batch} takes {sec.splitlines()[-1].split()[1]} sec"
            )
            # print(f'    one epoch for batch size: {batch} \n{sec}')
            time.sleep(0.5)


def iter_hidden() -> None:
    for train in train_size:
        print(f"train size: {train}")
        for hidden in range(50, 601, 50):
            sec = os.popen(
                f"echo -1 | ./bin/cuda -b 1200 -h {hidden} -t {train}"
            ).read()
            print(
                f"    one epoch for hidden size: {hidden} takes {sec.splitlines()[-1].split()[1]} sec"
            )


def main() -> None:
    os.system("make")
    # iter_batch('singleThread')
    iter_batch("openmp")
    # iter_batch('cuda')

    # iter_hidden()
    # draw_line_chart()


if __name__ == "__main__":
    main()
