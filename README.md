# spatial_filtering-CNN

## 공간 필터

흰색: 작은 값, 검정색: 큰 값 <br/>
myfil1=np.array([[1,1,1],[1,1,1],[-2,-2,-2]],dtype=float)  <br/>
myfil2=np.array([[-2,1,1],[-2,1,1],[-2,1,1]], dtype=float)  <br/>

myfil1에 대해서는 가로 라인의 아래쪽이 큰 값이 되고, myfil2에 대해서는 세로 라인의 왼쪽이 큰 값이 된다.<br/>

필터의 모든 요소를 합하면 0이 되도록 디자인되어 있으므로 공간 주고자 없는 균일한 부분은 0으로 변환되고,<br/>
필터에서 추출하려는 구조가 존재할 경우에는 0이상의 값으로 변환된다.(-값을 취하므로, 검정색으로!)

이미지의 일부분과 필터 요소를 곱한 합을 이미지를 slide 시키면서 이미지의 전 영역에서 구한다 -> **합성곱(convolution) 연산**

![image](https://user-images.githubusercontent.com/24853452/103439369-ad7add00-4c7f-11eb-998a-84aeb3f77182.png)

필터를 적용하면 출력 이미지의 크기가 작아지므로, 연속으로 다양한 필터를 적용하면 이미지가 점점 작아져 버린다.<br/>
이 대응책으로 padding이라는 방법이 있다. <br/>
padding은 필터를 적용하기 전에 0 등의 고정된 요소로 주위를 부풀려두는 방법이다.<br/>
필터는 한 칸씩 이동할 수도 있지만, 2칸이나 3칸 등 어떤 간격이든 이동할 수 있다. 이 간격을 stride라고 한다. <br/>


padding과 stride 값은 라이브러리로 합성곱 네트워크를 사용할 때, 인수로 전달

## 합성곱 신경망
필터를 사용한 신경망을 **합성곱 신경망(Convolution Neural Network: CNN)** 이라고 한다. <br/>

정답률은 무려 97.5%!
![image](https://user-images.githubusercontent.com/24853452/103440434-c6d45700-4c88-11eb-910b-5d33837a16c3.png)

![image](https://user-images.githubusercontent.com/24853452/103440712-47945280-4c8b-11eb-8215-5501d06939f3.png)

학습으로 얻은 필터는 다음과 같다.<br/>
3번 필터는 세로줄의 왼쪽 엣지를 강조!

![image](https://user-images.githubusercontent.com/24853452/103442186-4c5f0380-4c97-11eb-8272-6fedbc027b02.png)

## 풀링
이미지의 위치의 어긋남에 대한 견고성을 위해 -> **풀링** <br/>
![image](https://user-images.githubusercontent.com/24853452/103442335-a01e1c80-4c98-11eb-9e32-a174557379f4.png)

## 드롭아웃
학습 시에 입력층의 유닛과 중간층 뉴런을 확률 p로 임의로 선택하여, 나머지를 무효화하는 방법 <br/>
미니 배치마다 뉴런을 뽑아 다시 이 절차를 반복한다. <br/>
학습 후, 예측하는 경우에는 모든 뉴런이 사용된다.

![image](https://user-images.githubusercontent.com/24853452/103442580-c9d84300-4c9a-11eb-9358-c3722405adde.png)

학습 시에는 p의 비율의 뉴런밖에 존재하지 않는 상태에서 학습 -> 예측 시에는 전체 참가하여 출력이 커져버림 <br/>
예측 시에는 드롭아웃을 한 층의 출력 대상의 가중치를 p배로 하여 작게 설정하여 계산을 맞춘다. <br/>
드롭아웃은 여러 네트워크를 각각 학습시켜 예측 시에 네트워크를 평균화해 합친다

pooling, dropout을 모두 넣은 CNN -> 정확도 99.2%
![image](https://user-images.githubusercontent.com/24853452/103443150-9946d800-4c9f-11eb-8da9-0747b7d52904.png)
![image](https://user-images.githubusercontent.com/24853452/103443162-ab287b00-4c9f-11eb-97ca-57c63c7704a5.png)
