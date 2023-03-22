# [NTIRE Real-Time Super-Resolution](https://cvlai.net/ntire/2023/) @ CVPR 2023




### Evalutation Track1(x2)

```
python demo/runtime_demo.py --submission-id 1 --model-name repnet  --scale 2 --fp16
```
```
python demo/sr_demo.py --submission-id 1 --model-name repnet --checkpoint demo/model_zoo/repnet_x2.pth --scale 2 --lr-dir ./testset/LR2/ --save-sr --fp16
``` 

### Evalutation Track1(x3)

```
python demo/runtime_demo.py --submission-id 1 --model-name repnet  --scale 3 --fp16
```
```
python demo/sr_demo.py --submission-id 1 --model-name repnet --checkpoint demo/model_zoo/repnet_x3.pth --scale 3 --lr-dir ./testset/LR3/ --save-sr --fp16
``` 


## References

[1] [Mobile AI & AIM 2022 Challenge: Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs](https://arxiv.org/pdf/2211.05910.pdf)

[2] [NTIRE 2022 Efficient Super-Resolution Challenge](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Li_NTIRE_2022_Challenge_on_Efficient_Super-Resolution_Methods_and_Results_CVPRW_2022_paper.pdf)
