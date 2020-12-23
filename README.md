## BERT와 ELECTRA 이용한 자연어처리
#BERT와 koELECTRA 활용하여 네이버 영화리뷰 긍정/부정 분석#  
#BERT와 ELECTRA 활용하여 freinds 대사 감정 분석#   

코드 참고 사이트  
https://www.secmem.org/blog/2020/07/19/Sentiment-Analysis/  
http://aidev.co.kr/chatbotdeeplearning/8709  
https://github.com/monologg/KoELECTRA  
https://github.com/monologg/KoCharELECTRA  
https://heegyukim.medium.com/huggingface-koelectra%EB%A1%9C-nsmc-%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-1a23a0c704af  
https://github.com/Parkchanjun/KU-NLP-2020-1/blob/master/%5B5%5D%20Transformer%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D_%ED%95%9C%EA%B5%AD%EC%96%B4.ipynb    
https://dsbook.tistory.com/63    
https://github.com/jiwonny/nlp_emotion_classification/blob/master/friends_electra.ipynb  
https://m.blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221434157182&proxyReferer=https:%2F%2Fwww.google.com%2F  
https://wikidocs.net/44249  

## 실행환경
클라우드 서버와 GPU를 제공받는 google colab 에서 진행  
런타임 유형 하드웨어 가속기 : GPU Tesla V100-SXM2  

## Dataset information 
Naver 영화 리뷰는 git clone https://github.com/e9t/nsmc.git 에서 다운받아 이용했습니다.   
nsmc/ratings_train.txt를 train set으로 nsmc/ratings_test.txt를  test set으로 설정했습니다.  
영화리뷰는 긍정/부정에 따라 라벨이 주어집니다.   
Freinds 대사 감정분석 모델의 경우 'frineds_train', 'frined_dev' 를 train 으로 'frineds_test' 를 test set 으로 설정했습니다.   
각 데이터셋의 경우 발화(utterance) (최대길이 = maxlen) 와 그에 해당하는 감정 라벨이 주어집니다.  

## Requirements  
numpy  
pandas  
scikit-learn  
matplotlib  
nltk  
keras with TensorFlow backend  
transformers (for BERT, koElectra, Electra model)  
torch (for BER0, koElectra, Electra model)  

1. 리스트1
# 10. 코드블럭 추가하기

```swift
public struct CGSize {
  public var width: CGFloat
  public var heigth: CGFloat
  ...
}
```
