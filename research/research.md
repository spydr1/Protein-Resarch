# Introduction 
Convoultional neural network (CNN) 을 이용한 distance map 예측.  

## Distance map prediction.
### Input
input으로 주어진 fasta file로 부터 multiple sequence alignment (MSA)을 실행한다.  
MSA의 결과를 direct coupling analysis (DCA)을 이용해 [seq_length,seq_length,441] 개의 행렬로 변환한다.   
DCA는 i, j 번째에 어떠한 아미노산이 등장하는가에 대한 확률값이다.  
즉, i,j 번째 의 아미노산이 (AA, AB .. --)일 확률을 의미한다. 아미노산의 종류는 20개 이지만 "-" (gap)을 나타내는 
경우가 추가 되어서 총 21x21 = 441개의 확률값으로 나타난다.  

* PSSM : 아미노산(20) + gap(1) = 21
* One-hot sequence : 아미노산(20)  
* 1D feature = pssm (21) + One-hot sequence (20) + Positional Entropy (1) -> 총 42개
* DCA : (21x21 = 441)
* Positional Entropy : (1)
* 2D feature = DCA (441) + Positional Entropy (1) -> 총 442개  
   
-> 1D feature를 2D feature로 변환 ([예시]) -> concat ( i 번째 1d feature, j번째 1d feature) 의 형태로 사용한다.

[예시]: https://ars.els-cdn.com/content/image/3-s2.0-B9780128160343000079-f07-06-9780128160343.jpg

### Output
주어진 input으로 output distance map (2~18, 16 bins) 의 형태로 예측한다. ([distogram])  
output shpae : [seq_length, seq_length, 16]   

[distogram]: https://lh3.googleusercontent.com/proxy/FJiDahkyzmqGiSNeLskj-OHWYoVHkTb8A2-F22RqX1hEARvCDpRyZ2JlP4SNkPBOM-Od9Ps3REsA7tSLYTQuSKbJ8_gR8ea4OGBwE7xExFVIImsRy5gR4BYZc6Ru2A7fDTbagtTXpu2c1RpRxXC6FC6xfWqtt_qDcMNu25SYZj2s6rc


### Training  

#### Architecture  
<!-- architecture/respre.py -->
trRosetta - output is different. now, I use only distance or contact map.

<!-- B.Baker -->
아래의 B. Baker 를 확인. 

    

#### Train

##### Data
1. [Dunbrack](https://github.com/spydr1/worksheet/blob/master/experiment/jbc/PDB70.md) # todo 링크 
    i) has template(s) not too far or close in sequence space;  
    ii) does not have strong contacts to other protein chains,  
    iii) should contain minimal fluctuating (i.e. missing density) regions.  
    iv) sequence similarity 40%  
    v) Minimum resolution 2.5 Å  
    vi) limiting their size to 50-300 residues
   
2. MSA의 결과가 최소 100개 이상인 경우만 사용

3. Final list 
1. Dunbrack & 2. MSA 결과가 100개 이상인 경우의 교집합     
Totally, number of data : 9062 (나의 부등호 실수로 인해서 일단 단백질 길이가 201~299인 데이터로 사용)

5. Data split : train - 8000, validation - 1062
데이터가 너무 커서 여러개의 파일로 관리하고 있고 이 경우 split 하기 어려워서 1000개의 단백질의 정보를 하나의 데이터 파일로 만든다.
   총 10개의 데이터 파일이 생성되고   
* 파일1~8 -> train
* 파일9~10 -> validation  
으로 나누었다.


##### Hyper parameter
<!-- todo optimizer 링크 -->
1. Adam optimizer 
2. Loss function : Sparsecrossentropy
3. Batch size 6 (GPU당 배치 1)
4. lr : 1e-2 ~ 1e-7 (PolynomialDecay)
5. step : 1000 
6. epoch : 1000   
-> 일반적으로 말하는 epoch의 경우 : 750 epoch 
7. Warm up step : 10000
8. weight decay : 1e-5

## Folding
<!-- 폴딩이 왜 필요한가 --> 
distance map을 3d 구조로 변환해야한다.   
distance map이 정확하면 비교적 쉽게 3d 구조를 만들수 있다. 방정식으로서 정확히 3d 구조의 좌표값을 알아내는게 가능한지는 정확하게 모르겠다.  
아마 초기값과 각도에 대한 제약조건을 주면 가능할수도 있겠다.  
어쨌거나 iterative하게 가능하다. (MDS) distance map만을 이용해서 3d 구조를 바꾸면 사실 가능한 구조는 무한개이다.  
그 이유는 회전에 대한 제약이 없어서이다. 또 특히 혹은 대칭적 (chirality)인 상황을 생각 해봐야 하는데 이는 biological하게 제외 시킬수 있다. (아마도?)  
-> 점으로 이루어진 어떤 특정한 구조를 상상하고 distance map을 구해보고, 이 구조를 회전변환 했을때 distance map 바뀌는지 생각해보면 알 수 있다.    
```
x + y = 2
2x- 5y = 5
매개변수가 두 개 있을때 방정식이 두개 있으면  풀 수 있지만

x + y = 1~3
2x- 5y = 4~6 
형식일때 어떻게 풀수 있을까 ? (1~3이라고 간략하게 표기했지만 일단은 정규분포라고 가정해보자)  
``` 
하지만 현재의 distance map prediction은 확률값으로 존재하고 있으므로 더욱 어려운 과제가 된다.
그런 것들을 극복하기 위해서 제약이 있어야 하고 가능한 제약은 그래도
1. distance map을 최대한 따르게 한다.  
2. 예측된 secondary structure 따르게 한다. (왜냐하면 ramachandran distribution을 알고 있으니까 이는 phi, psi에 대한 제약을 의미한다.) 
3. Steric violation - Alpha fold2에서 언급된 적이 있으므로 자세히 확인해보자
4. Van der waals  
그 외에 뭐가 있는지 잘 모르겠다. 이는 매우 biological한 정보이니까 내가 다 알수는 없을것 같다. 
   이거는 그래서 고전적인 단백질 구조를 예측하는 그룹의 것들을 많이 봐야 할것 같고 이러한 부분은 
   딥러닝으로 해결할수 없고, 협업이 필요하다. 
   
일단 현재의 경우는 DConstruct라는 open source 프로그램으로 3d structure로 예측하고 있다.   
% Secondary structure를 알고 있다고 가정하고 풀었다.  
-> 현재의 모델로도 secondary structure를 예측 할수 있다고 생각하지만   
1. 추후 Attention 계열의 모델로 ss 부분 예측을 대체할 예정이므로 추가적인 작업을 할 예정이 없다.  
2. CNN이 적합한 방식이라고 생각하지 않는다. (주관적)  
-> CNN의 모호한 receptive field이 문제라고 생각하고, 적어도 attention score를 확인 할 수 있는게 타당하다고 생각한다.
    
## Additional experiment plan
###### 이라고 적지만 실제로 할 계획은 없다.

### variable length 
* 현재 모델은 학습은 input shape = (300,300,feature dimension)을 가정하고 학습한다.   
50 ~ 300길이를 가지는 단백질에 대해 학습하였고 crop의 과정을 추가하면 모든 길이에 대해 대응 할 수 있다.    
-> Related work   
Baker, Alphafold 모두 crop의 과정을 겪고 있으며 각각 input shape = (300,300,feature dim), (64,64,feature dim)이다.  
Alphafold의 경우 baker 그룹과 다르게 input shape 굉장히 줄어들었기 때문에 훨씬 더 큰 모델을 학습 시킬수 있었다.   
더 큰 모델이라고 하기 조금 애매하지만, 더 많은 파라미터를 가진 모델 <- 파라미터 갯수를 확인해보진 않았다.    
    

* 간략하게 모델의 layer를 비교해보면 : 61 residual blocks (baker) vs  220 residual blocks (alphafold1)
* intuition 1 : Alphafold는 이러한 crop training이 augmentation효과를 낸다고 이야기한다.  
-> thought 1. receptive field를 생각해보자  
-> thought 2. Alphafold의 경우 명확하게 64 x 64의 영역 까지만 고려하고 있으며 이는 전체구조를 고려하는 것은 아니다.    

* intuition 2 : Alphafold는 거리가 22보다 큰 정확하게 예측하기 어렵다고 이야기 하고 있음.    
1. 2-20Å into 37 equal bins. (baker)  
2. 2-22Å into 64 equal bins. (Alphafold)  
3. 2-17Å into 16 equal bins. (ours)  
-> 나의 생각 또한 거리가 멀다라는 것은 상호작용이 없음을 의미하기 때문에 이 경우에는 예측하는 것이 불가능하다고 생각한다.  
-> contact예측, 여러가지 제약조건에 의해서 long distance case가 결정되어 질수있다고는 생각한다.   
   
## data 
1. fasta의 길이가 50~300 까지만 학습했다.  
2. msa 결과가 50개 이상일때만 학습했다.  
3. msa 결과가 매우 많은 경우도 있지만 200까지만 잘라썼다.

## More layer 
* Model 자체의 변경.



* Ref - Section Distogram prediction & Section Neural network hyperparameters in Alphafold
## Train detail 

### A. Alphafold
* 7 groups of 4 blocks with 256 channels, cycling through dilations 
1, 2, 4, 8.
* 48 groups of 4 blocks with 128 channels, cycling through dilations 
1, 2, 4, 8.
* Optimization: synchronized stochastic gradient descent
* Batch size: batch of 4 crops on each of 8 GPU workers.
* 0.85 dropout keep probability.
* Nonlinearity: ELU.
* Learning rate: 0.06.
* Auxiliary loss weights: secondary structure: 0.005; accessible surface area: 0.001. These auxiliary losses were cut by a factor 10 after 
100 000 steps.
* Learning rate decayed by 50% at 150,000, 200,000, 250,000 and 
350,000 steps.
* Training time: about 5 days for 600,000 steps.

* Ensemble : predictions from an ensemble of four separate models, 
  trained independently with slightly different hyperparameters,
  
### B. Baker
* output : distance histogram (d coordinate) and 3 angle histograms (ω, θ
and φ coordinates)
  
* 61 groups of 2 block with 64 channel, cycling through dilation 1, 2, 4, 8, 16
* 0.85 dropout keep probability.
* Nonlinearity: ELU.
* Learning rate: 0.0001.
* weight decay : 0.0001
* Training time: about 9 days - one NVIDIA Titan RTX GPU (single network)
* Ensemble : We train 5 networks with random 95/5% training/validation splits and 
  use the average over the 5 networks as the final prediction.
  
   
## Discussion

### 상삼각행렬에 대해서만 loss를 계산하고 있다. (대각행렬 제외)  
baker 그룹에서는 (상삼각행렬 + 하삼각행렬)/2 의 형태로 계산하는 것으로 보이는데 내 생각은 상삼각행렬만 계산하는게 맞는 것 같다. 
  3x3 convolution 하나의 채널만 있다고 가정한 경우 하삼각행렬에 속하는 영역을 계산할때 상삼각행렬에서 계산된 것과
  완전히 전치된 값이 들어 올텐데 그러한 상황이 별로 타당해 보이지 않는다. 하삼각행렬은 상삼각행렬의 전치행렬일 뿐이지만 마치 
  두개의 상황이 있고 둘을 모두 예측해야 하는 상황이 될 것 같다. 이는 쓸데 없이 모델의 복잡도를 늘려야 하는 경우로 나타날 것이라고 판단했다.  


```
ex)
1 2 3  
3 4 5  x Convolution Layer = output1 
5 6 7  

1 3 5  
2 4 6  x Convolution Layer = output2 
3 5 7

output1 = output2 가 되려면  Convolution Layer가 어떤 값을 가져야 할까
```  
-> 만약 의미가 있을 경우가 있다면 AA의 순서를 정방향, 역방향을 정할 수 없을 때  
-> 그러나 COOH가 끝나는점에있는게 C-ter고 N-ter는 COOH가 다음에 오는 아미노산이랑 연결되있음.  
-> 순서가 반대일 경우는 생각하지 않아도 된다.  (ref @조병철)

* 상삼각행렬 중에서도 mask를  j-i>x. x의 값을 몇으로 두는게 나을지 생각해봐야한다. i, i+1 번째의, 즉 이웃한 아미노산의 경우 CA의 거리는 거의 3.8Å으로 고정이다.
인덱스의 거리가 가까울수록 자유도가 낮다. medium-range, long-range contact만 중요하다고 가정한다면 굳이 short-range에 대한 loss를 계산할 필요가 있을까 ?
  medium-range, long-range를 잘 구하고 나면 short-range의 경우에는 자명하게 구조가 정해질수도 있다. 
  
* over-fitting 
1. 해당 문제의 경우 오버피팅이 정말 좋지 않은 것일까에 대한 생각
2. MSA의 결과를 몇개 써야 할까 

### number of MSA result 
* cutted MSA result : 현재 200까지만 잘라서 쓴다. 어떤 경우에는 1000개씩 나오기도 한다. 상위의 200개만 잘라서 쓴다는 것은 타겟 단백질과 강하게 연관되어 
  있는 것들로만 고려한다는 의미이다. 이는 casp14에 baker, alphafold에서 언급한것과 상반된다고 받아들여진다. baker는 pixel-wise Attention에 
  대한 언급을 하며 몇개의 MSA 결과, 어떤 분포로 MSA결과를 가져올지에 대한 실험을 했다. 
  유사도가 높은것만을 가져왔을 때 정확도 높다고 보장할수 없고, 각 유사도별 대표되는 것들을 가져오는게 중요할지도 모른다. 왜냐하면
  타겟과 유사도가 높은것들을 다시 군집화 했을때 10개의 군집이 있다고 가정 할수 있고 그중 대표되는 10개의 단백질의 조합으로 모양이 
  타겟 단백질의 모양일 가능성도 있다. 유사도란것이 전체를 봤을때 유사한 것을 기준으로 점수를 매긴것이고 부분적으로는 10개의 군집중 
  유사도가 가장 떨어지는 것이 부분적으로 유사도가 높았을수도 있다. 또한 이 내용이 alphafold2가 언급하는 것과 비슷한것 같기도 하다. 
  PPT에서 그림을 보면 MSA결과를 놓고 종별로 비슷한 것들을 몇가지 발췌해오는 것으로 보인다.
  이것이 타겟과 유사도가 높은것들을 다시 군집화 했을때 같은 종에 해당하는 것이 10개의 군집으로 나타나지는 것이고 그 군집중에 대표되는 것들을 뽑는 것이
  일부가 아닌 전체를 관찰하는 것으로 나타날수 있다. ([d14])  
-> intuition 1.  
현재 타겟 단백질이 인간의 것이라고 가정했을 때, 유사도가 높은 것들을 가져오는 행위는 인간종에서만 비슷한 단백질을 가져 오겠다라는 것과 같을수 있으며
이는 타겟 단백질과 닮았을 것이라고 무조건적으로 보장할 수 없다. 오히려 다른 종이 인간의 단백질에게 무언가를 상호작용 하기 위한 무기로서 단백질의 구조를 결정하였다면
아직은 알수 없는 방패의 모양을 무기의 모양을 보고 유추 할수도 있다.

[d14]: https://github.com/spydr1/worksheet/discussions/14




## Future
1. pdb 구조 없더라도 sequence로만으로도 학습 할수 있게끔 하자.  
   -> masked language model (MLM)   
   -> bert에 관한 연구들이 이미 여럿 존재한다. ( facebook esm)  
2. ss 예측  
3. end-to-end 예측    
a. CNN distance map predictor + Attention ss predictor -> 3d structure  
b. Attention ss + Attention distance map predictor  -> 3d structure  
   -> 특히 bert를 이용하여 sequence를 embedding 할수 있고, 이는 MSA -> DCA와 유사한 방법이라고 생각한다.   
   -> poor information -> rich information으로 바꾸어 주는 방법 중에 하나라고 생각한다.   
   
4. Folding & Refinement
5. Ensemble
6. Missing 처리
* fasta에서 X로 표시되고 이걸 빼놓고 msa든 뭐든 하는거 같은데 다른건 다른것이다.
7. loss 값과 contact evaluation은 다르게 평가 되어진다. 
   학습값은 2~3, 3~4 ,4~5 .. 불연속적인 거리값을 구하는 것이고 contact자체를 의미하는건 아니다.
   그래서 loss 값이 높아져도 contact precision은 높을수 있다.
   그리고 7이라고 예측했을때 6인 경우에 대한 (꽤 잘 맞췄지만 ?) loss 값의 weight를  조정 해줄 필요가 있다. 
   현재는 bins 형태로 되어있으니까 ..? real value 로 보면 사실상 크게 틀리지 않았다. 
   bins로 불연속적으로 나타내니까 정답이 6이고, 7이라고 예측한 경우, 15라고 예측한 경우 loss 값의 크기는 같다.
   (잘못 예측한 정도의 확률값 (p 값, 혹은 softmax output) 은 같다고 했을때)
 
<!-- 
1. dunbrack 
2. baker list 
3. msa
4. hhblits
5. ref
-->