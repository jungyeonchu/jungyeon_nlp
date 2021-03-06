# BERT와 ELECTRA 이용한 자연어처리
#BERT와 koELECTRA 활용하여 네이버 영화리뷰 긍정/부정 분석#  
#BERT와 ELECTRA, koELECTRA 활용하여 freinds 대사 감정 분석#   

모든 코드는 아래 사이트를 참고하였습니다   
코드 참고 사이트  
https://www.secmem.org/blog/2020/07/19/Sentiment-Analysis/  
http://aidev.co.kr/chatbotdeeplearning/8709  
https://github.com/monologg/KoELECTRA  
https://github.com/monologg/KoCharELECTRA  
https://heegyukim.medium.com/huggingface-koelectra%EB%A1%9C-nsmc-%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-1a23a0c704af  
https://github.com/Parkchanjun/KU-NLP-2020-1/blob/master/%5B5%5D%20Transformer%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D_%ED%95%9C%EA%B5%AD%EC%96%B4.ipynb     
https://github.com/cjmp1/nlp-sentiment-analysis/tree/master/ENG  
https://github.com/jiwonny/nlp_emotion_classification/blob/master/friends_electra.ipynb    
https://m.blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221434157182&proxyReferer=https:%2F%2Fwww.google.com%2F    
https://wikidocs.net/44249    
https://dsbook.tistory.com/63    

## 실행환경
클라우드 서버와 GPU를 제공받는 google colab 에서 진행  
런타임 유형 하드웨어 가속기 : GPU Tesla V100-SXM2  

## Dataset information 
Naver 영화 리뷰는 git clone https://github.com/e9t/nsmc.git 에서 다운받아 이용했습니다.   
nsmc/ratings_train.txt를 train set으로 nsmc/ratings_test.txt를  dev set으로 설정했습니다.  
영화리뷰는 긍정/부정에 따라 라벨이 주어집니다.   
Freinds 대사 감정분석 모델의 경우 'frineds_train', 'frined_test' 를 train 으로 'frineds_dev' 를 dev set으로 설정했습니다.   
Freinds 데이터셋의 경우 발화(utterance) (최대길이 = maxlen) 와 그에 해당하는 감정 라벨이 주어집니다.  

## Requirements  
re  
json  
numpy    
pandas    
scikit-learn    
matplotlib  
nltk  
keras with TensorFlow backend  
transformers (for BERT, koElectra, Electra model)  
torch (for BERT, koElectra, Electra model)  

## 사용 모델  
다양한 모델을 확보한 뒤 voting을 통한 결과 예측을 위해 여러 case별 모델을 사용함  
### 네이버
2가지 모델   
bert-base-multilingual-cased   
monologg/koelectra-base-v3-discriminator(데이터정제 유/무)    
로 총 3개   
### Friends 
4가지 모델   
bert-base-multilingual-cased  
bert-base-uncased  
monologg/koelectra-base-v3-discriminator  
google/electra-large-generator(데이터정제 유/무)      
로 총 5개  

## 실행방법

## 네이버 영화 리뷰 실행 
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
   
 5. 구해진 token을 숫자 값으로 indexing 하고, maxlen을 이용해 padding 진행, 그리고 attention_masks를 설정   
 6. 학습을 위해 torch tensor 형태로 모든 데이터들을 변환  
 7. 4,5작업 test데이터에 동일 반복  
 8. labels에 train data의 label을 저장  
 9. GPU 이용 가능 확인(코랩 GPU 이용)  
 10. pretrained 된 모델을 model에 불러오기
   ```
   # 분류를 위한 BERT 모델 생성
   model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
   model.cuda()
   ```
   ```
   #분류를 위한 koelectra_v3 모델 생성
   model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
   model.cuda()
   ```   
11. optimizer, epoch등 하이퍼파라미터, scheduler등 설정  
12. Training,validation 진행  
13. kaggle 데이터를 불러온 후 정제  
14. kaggle 데이터 예측실행 함수 정의  
15. kaggle 데이터 예측   
16. kaggle 데이터 예측 결과 dataframe 생성  
17. 모델별 실행한 데이터 예측 결과 csv 다운받아 엑셀에서 hard voting 
![image](https://user-images.githubusercontent.com/76143729/103061037-5b85e780-45ed-11eb-964f-cccb8e21df9e.png)   

18. hard voting 예측 데이터 업로드   

## 한국어 kaggle 예측 결과 

![image](https://user-images.githubusercontent.com/76143729/103059379-48244d80-45e8-11eb-9847-347c949a2937.png)  

## friends  실행  
1. 필요한 package 모두 import  
2. kaggle 연동하여 test 데이터 불러오기 
3. friends.json데이터 불러와서 train, test 데이터는 train_data에 저장, dev 데이터는 dev_data저장 kaggle 불러온 데이터는 test_data에 저장  
4. train 데이터 전처리 함수 정의 
   라벨추출, bert 또는 Electra tokenizer를 사용해 토큰으로 분리하기 위해 문장 편집  
   어텐션 마스크 초기화 및 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정 
   ```
   #라벨추출 
   labels = data['emotion'].values
   encoder.fit(labels)
   labels = encoder.transform(labels)
   #electra large로 토크나이징
   tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
   tokenized_texts = [tokenizer.tokenize(utterance) for utterance in utterances]
   # 어텐션 마스크 초기화
   attention_masks = []
   # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
   for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)
   ```
5. test 데이터 전처리 함수 정의 
   라벨추출, bert 또는 koelectra tokenizer를 사용해 토큰으로 분리하기 위해 문장 편집  
   어텐션 마스크 초기화 및 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정  
   
6. labels에 train_data의 label을 저장, 학습을 위해 torch tensor 형태로 모든 데이터들을 변환
7. GPU 이용 가능 확인(코랩 GPU 이용)  
8. pretrained 된 모델을 model에 불러오기
   ```
   # BERT 모델 생성
   model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=8)
   model.cuda()
   ```
   ```
   #electra-large-generator 모델 생성
   model = ElectraForSequenceClassification.from_pretrained('google/electra-large-generator', num_labels=8)
   model.cuda()
   ```
   ```
   #koelectra base 3 모델 생성-한국어 pretrained된 모델의 영어 적용 테스트
   model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",num_labels=8)
   model.cuda()
   ```   
9. optimizer, epoch등 하이퍼파라미터, scheduler등 설정  
10. Training,validation 진행     
11. kaggle 데이터 예측실행 함수 정의  
12. kaggle 데이터 예측   
13. kaggle 데이터 예측 결과 dataframe 생성  
14. 모델별 실행한 데이터 예측 결과 csv 다운받아 엑셀에서 hard voting  
![image](https://user-images.githubusercontent.com/76143729/103133943-d08e1580-46f0-11eb-81b9-17635096bf56.png)   
15. hard voting 예측 데이터 업로드  

## 영어 kaggle 예측 결과  

![image](https://user-images.githubusercontent.com/76143729/103133975-0b904900-46f1-11eb-95c7-510b4428d651.png)







