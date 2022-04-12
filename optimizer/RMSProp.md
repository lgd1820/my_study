# RMSProp

![](https://render.githubusercontent.com/render/math?math=h_i%20=%20ph_{i-1}%20%2b%20(1-p)\frac{\partial%20L_i}{\partial%20W}*\frac{\partial%20L_i}{\partial%20W})  

![](https://render.githubusercontent.com/render/math?math=W=%20W%20-\eta\frac{1}{\sqrt{h}}\frac{\partial%20L}{\partial%20W})


AdaGrad의 단점을 보완한 기법으로 AdaGrad는 과거의 기울기를 제곱하여 더하는 방식으로 진행하기 때문에 점점 갱신 강도가 약해진다는 단점이 있다. RMSProp에서는 제곱의 합이 아니라 지수평균으로 바꾸어 먼 과거의 기울기는 조금 반영하고 최근 기울기를 많이 반영하는 지수이동평균(Exponential Moving Average)을 썼다.
