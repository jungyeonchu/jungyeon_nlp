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


## 실행방법

# 네이버 영화 리뷰 실행 
1. 필요한 package 모두 import  
2. 데이터 불러와서 raitings_train은 train, ratings_test는 test 데이터로 저장
3. CleanText함수로 데이터 정제  
4. 정제된 문장을 bert 또는 koelectra tokenizer를 사용해 토큰으로 분리하기 위해 문장 편집   
    앞에 [CLS] , 뒤에 [SEP] 을 달아줌. cls : classification , sep : 문장 구분 
   
   ```
   #모델에 맞게 형식 변환  
   sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in clean_sentence]
   ```
 
   버트 토크나이저 실행
   
   ```
   #bert_base-mulmultilingual-cased 토크나이저 실행 
   tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
   tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
   ```
   
   코일렉트라 토크나이저 실행
   
   ```
   # koelectra-base-v3-discriminator토크나이저 실행
   tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
   tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
   ```


json 파일 읽은 후, 3파일 모두 cleaning 함수를 통해 아래 과정 진행 후 train, dev는 train_data에 저장 test는 test_data에 저장
영어 이외 data re 패키지를 이용해 제거
소문자로 모두 통일
nltk 의 stopwords를 이용해 불용어 제거
nltk 의 stemmer를 이용해 stemming
구해진 문장을 bert tokenizer를 사용해 토큰으로 분리하기 위해, 문장 편집 (앞에 [CLS] , 뒤에 [SEP] 을 달아줍니다. cls : classification , sep : 문장 구분) 후, token으로 분리
3번에서 구해진 token을 숫자 값으로 indexing 하고, maxlen을 이용해 padding 진행, 그리고 attention_masks를 설정 (데이터가 >0 인 단어부분에 attention을 주어서 학습 속도와 성능을 향상시킵니다.)
labeltoint 함수를 생성해 labels에 train_data의 label을 저장, 학습을 위해 torch tensor 형태로 모든 데이터들을 변환해줍니다.
cell3~cell5에 해당하는 과정을 test_data에 대해서도 진행해줍니다.
GPU 사용가능 여부를 확인(colab의 경우 가능) 후, pretrained 된 모델을 model에 불러오고, optimizer와 각종 파라미터, scheduler들을 세팅
training 진행
kaggle 데이터를 불러온 후 제출을 위해 dataframe 생성하는 부분
실제 문장 확인하는 구문 predict('') 안의 내용을 원하는 문장으로 바꿔주면 됩니다.


1. 리스트1
# 10. 코드블럭 추가하기

```swift
public struct CGSize {
  public var width: CGFloat
  public var heigth: CGFloat
  ...
}
```
