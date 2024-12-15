## MACRec: a Multi-Agent Collaboration Framework for Recommendation usin Contract Net Protocol with an LLM Evaluation

This repository contains an unofficial implementation of our SIGIR 2024 demo paper:
- [Wang, Zhefan, Yuanqing Yu, et al. "MACRec: A Multi-Agent Collaboration Framework for Recommendation". SIGIR 2024.](https://dl.acm.org/doi/abs/10.1145/3626772.3657669)





### Setup the environment

0. Make sure the python version is greater than or equal to 3.10.13. We do not test the code on other versions.

1. Run the following commands to install PyTorch (Note: change the URL setting if using another version of CUDA):
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
2. Run the following commands to install dependencies:
    ```shell
    pip install -r requirements.txt
    ```
3. Run the following commands to download and preprocess the dataset:
   ```shell
   bash ./scripts/preprocess.sh
   ```

### Run with the command line

Use the following to run specific tasks:
```shell
python main.py -m $task_name --verbose $verbose $extra_args
```

Then `main.py` will run the `${task_name}Task` defined in `macrec/tasks/*.py`.

E.g., to evaluate the sequence recommendation task in MovieLens-100k dataset for the `CollaborationSystem` with *Reflector*, *Analyst*, and *Searcher*, just run:
```shell
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/reflect_analyse_search.json --task sr
```

You can refer to the `scripts/` folder for some useful scripts.

### Run with the web demo

Use the following to run the web demo:
```shell
streamlit run web_demo.py
```

Then open the browser and visit `http://localhost:8501/` to use the web demo.

Please note that the systems utilizing open-source LLMs or other language models may require a significant amount of memory. These systems have been disabled on machines without CUDA support.

### Citation
If you find our work useful, please do not save your star and cite our work:
```
@inproceedings{wang2024macrec,
  title={MACRec: A Multi-Agent Collaboration Framework for Recommendation},
  author={Wang, Zhefan and Yu, Yuanqing and Zheng, Wendi and Ma, Weizhi and Zhang, Min},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2760--2764},
  year={2024}
}
```
