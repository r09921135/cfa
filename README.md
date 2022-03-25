# Domain Adaptation with Contextual Foreground Attention for Action Segmentation

## Contribution [[Slides]](https://github.com/r09921135/cfa/blob/master/(Slides)CFA.pdf)
* Propose the Contextual Foreground Attention for domain adaptation of action segmentation. 
* Design an attention mechnism to capture the temporal relations of actions.
* Utilize foreground labels to filter out the irrelavent information.
* Outperm the existing frame-wise DA approaches and achieve state-of-the-art in GTEA dataset.

## Results
### GTEA dataset
|    Methods   |  F1@10 | F1@25 |   F1@50  |   Edit  |  Acc |   
| :------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|    [ASRF](https://paperswithcode.com/paper/alleviating-over-segmentation-errors-by)   |   89.4   |    87.8    |   **79.8**   |   83.7  |   77.3  |
|    [ASFormer](https://paperswithcode.com/paper/asformer-transformer-for-action-segmentation)   |   90.1   |    88.8    |   79.2   |   84.6  |   79.7  |
|    [SSTDA](https://paperswithcode.com/paper/action-segmentation-with-joint-self)   |   90.0   |    89.1    |   78.0   |  86.2  |   79.8  |
|     Ours     |   **91.2**   |    **89.4**    |   78.5   |   **87.0**  |   **80.0**  |

## Quick Run
* This implementation is modified from [SSTDA](https://github.com/cmhungsteve/SSTDA), where you can find more detail about usage.
* The script file of our Contextual Foreground Attention (CFA) version is in `scripts/run_gtea_CFA.sh`. You can run the script as below:
```
bash scripts/run_gtea_CFA.sh
```
The script will do training, predicting and evaluation for all the splits on the GTEA dataset.
