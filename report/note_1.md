Bu bizim deney için iyi, çünkü M1’in doğru/yanlış ayrımı rastgeleliğe bağlı olmasın. Eğer temperature açarsak aynı soru bazen doğru bazen yanlış çıkabilir.

do_sample=False
max_new_tokens=160
temperature: kullanılmıyor
top_p: kullanılmıyor

# Çözülemeyen soruları tespit için çalıştırdığım kod

python scripts/evaluate_m1_failures.py --batch-size 4 --max-new-tokens 256 

Çıktı olarak m1_predictions.jsonl dosyası üretildi. Bu dosyanın içeriği şöyle:

id                    -> örnek id'si, örn. train-0
source_dataset        -> ytu-ce-cosmos/gsm8k_tr
source_split          -> train
model_id              -> Qwen/Qwen3.5-4B
model_type            -> qwen3_5
question              -> modele sorulan soru
answer                -> dataset'ten gelen orijinal çözüm/cevap metni
reference_answer_raw  -> answer içinden yakalanan ham final sayı
reference_answer      -> normalize edilmiş doğru cevap
prompt                -> M1'e verilen tam prompt
model_output          -> M1'in ürettiği cevap
predicted_answer_raw  -> model_output içinden yakalanan ham final sayı
predicted_answer      -> normalize edilmiş model cevabı
is_correct            -> model cevabı referansla uyuşuyor mu?

Bu dosya içerisinden aynı formatta yanlış cevaplar train_failed.jsonl dosyasına alındı. Yanlış cevapların oranı şöyle:

Toplam değerlendirilen soru: 8768
Doğru cevap: 4683
Yanlış cevap: 4085
Accuracy: 0.5341
Atlanan referans parse edilemeyen soru: 24

Bu yanlış cevapların bazıları muhtemelen gerçekte yanlış değil. Çünkü referans cevap olarak aldığımız cevaplarda yanlış olanlar var. O yüzden bu yanlış yapılan soruları başka bir llm'e sorup ondan da cevap alacağım ve ondan gelen cevaplar ile referans cevap uyuşmasına rağmen qwen modelinin bu cevap ile uyuşmaması durumunda bu soruyu yanlış olarak kabul edeceğim. Bunun için openrouter ile gpt-oss-120b api sini kullanıp aynı prompt ile train_failed.jsonl dosyasındaki soruları gpt-oss-120b ye göndereceğim ve aynı dosya formatında ondan cevap alacağım. Bu sayade çözülemeyen soruların çözümünü başarılı bir modelden alma görevini de yapmış olacağız aslında. 

Bu işlem için python scripts/evaluate_openrouter_teacher.py i yazdık. 



# Test Değerlendirmesi

Base modelin test.jsonl veri kümesindeki sonuçları:

Model: Qwen/Qwen3.5-4B
Test soru sayısı: 500
Doğru: 290
Yanlış: 210
Accuracy: 0.58

> Doğrulanmış veri kümesi tamamlandıktan sonra test kümesindeki örneklerin içerisinde olmadığı en az 500 soru seçeceğim. Bu nihai 500 soru eğitim kümemizi oluşturacak.

Saat formatındaki cevaplar eğitim ve test kümesinden çıkartılıp sadece sayısal cevaplar olacak şekilde ayarlandı.



# Aşama 1 eğitimi

source .venv/bin/activate && python scripts/selective_loop.py --restart --train-epochs 3 --learning-rate 2e-4 --weight-decay 0.0 --max-grad-norm 1.0 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-train-tokens 1536 --max-input-tokens 1024 --max-new-tokens 256 --seed 42



# Sunum 

xdg-open /home/muhammet/Documents/Projects/Hesaplamali_Anlambilim_Final/sunum.html



