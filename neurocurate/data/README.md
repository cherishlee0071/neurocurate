# 重要！
## 由于这个笔记本磁盘空间不大，因此things-eeg数据集难以下载。
## things-eeg数据集储存在https://www.nemar.org/dataset/nm000232 你在访问需要的数据时，请遵循“最小本地磁盘占用原则（D盘只有200多G了，千万不要过度下载原始EEG数据！）”，推荐：
1.使用 EEGDash 进行流式读取（最推荐）：EEGDash 是一个 Python 库，它可以从 NEMAR 按需流式传输 EEG 数据，并直接返回为 PyTorch 或 braindecode 数据集，而无需将原始数据文件完整下载到磁盘
代码：
//安装依赖
pip install eegdash

from eegdash import EEGDashDataset

//指定一个小的本地缓存目录，仅缓存当前需要的分块数据
ds = EEGDashDataset(dataset="nm000232", cache_dir="./eeg_cache")

print(f"数据集包含 {len(ds)} 个 recordings")
//之后就可以像使用普通 PyTorch 数据集一样使用 ds，数据会在后台按需加载

2.使用 DataLad/git-annex 进行“按需获取”
你可以先 clone 仓库的“指针”元数据（几乎不占空间），然后只 get 你实际需要的那几个文件。
代码：
//克隆元数据（不包含数据内容，仅几百KB）
datalad clone https://github.com/nemarDatasets/nm000232 nm000232
cd nm000232

//2. 查看目录，确定你需要的具体文件（例如只下载 sub-01 的预处理数据）
ls sub-01/ses-01/eeg/

//3. 仅获取你指定的那一个文件
datalad get sub-01/ses-01/eeg/sub-01_ses-01_task-test_eeg.bdf