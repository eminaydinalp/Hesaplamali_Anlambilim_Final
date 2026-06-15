# LLM'lerde Matematiksel Problem Çözme için Farklı LoRA İnce Ayar Yaklaşımlarının Değerlendirilmesi

Muhammet Emin AYDINALP  
25501819

Sunum hedefi: 5-10 dakika  
Önerilen akış: 8 slayt, yaklaşık 7-8 dakika

---

## Slayt 1 - Başlık

**LLM'lerde Matematiksel Problem Çözme için Farklı LoRA İnce Ayar Yaklaşımlarının Değerlendirilmesi**

Muhammet Emin AYDINALP  
25501819

**Konuşmacı notu:**  
Bu çalışmada büyük dil modellerinin matematiksel sözel problemleri çözme başarısını, öğretmen modelden alınan çözümlerle yapılan LoRA ince ayar üzerinden inceledim. Asıl amaç, modelin yanlış yaptığı sorulardan öğrenmesinin benzer sorulara ve bağımsız test kümesine nasıl yansıdığını görmekti.

---

## Slayt 2 - Problem ve Motivasyon

- Matematiksel sözel problem çözme, LLM'lerin akıl yürütme kapasitesini ölçmek için sık kullanılan bir görevdir.
- Modelin yalnızca bilgi hatırlaması yeterli değildir.
- Problemdeki nicelikleri ayırması, ilişkileri kurması ve doğru hesap sırasını takip etmesi gerekir.
- Bu çalışmada odak soru şudur:
  **Modelin yanlış yaptığı bir soruyu öğretmen çözümünden öğrenmesi, benzer ve bağımsız sorulara aktarılabilir mi?**

**Konuşmacı notu:**  
Matematik problemlerinde doğru cevaba ulaşmak çoğu zaman birkaç adımlı muhakeme gerektiriyor. Bu yüzden modelin hatalı olduğu soruları öğretmen model yardımıyla düzeltmek ilk bakışta doğal bir çözüm gibi görünüyor. Fakat burada önemli soru şu: Öğrenilen bilgi sadece o soruya mı yarıyor, yoksa benzer ve bağımsız sorulara da aktarılıyor mu?

---

## Slayt 3 - Veri Seti ve Modeller

- Veri kümesi: **GSM8K_TR**
- Temel model: **Qwen/Qwen3.5-4B**
- Öğretmen model: **gpt-oss-120b**
- Sabit test kümesi: **500 soru**
- Eğitim kümesi: M1'in yanlış yaptığı ve öğretmen modelle doğrulanan **500 soru**
- Eğitim ve test kümeleri arasında çakışma yoktur.

**Konuşmacı notu:**  
Deneylerde Türkçe matematiksel sözel problemlerden oluşan GSM8K_TR kullanıldı. Önce temel modelin hatalı çözdüğü örnekler belirlendi. Bu hataların gerçekten öğretilebilir hata olup olmadığını kontrol etmek için öğretmen modelden doğru çözüm alındı. Daha sonra bu örneklerden 500 soruluk eğitim kümesi, ayrı olarak da 500 soruluk test kümesi oluşturuldu.

---

## Slayt 4 - Genel Deney Akışı

1. M1 modeli GSM8K_TR üzerinde değerlendirildi.
2. M1'in yanlış yaptığı sorular belirlendi.
3. Öğretmen modelden bu sorular için doğru çözüm alındı.
4. Öğretmen modelden aynı yapıda benzer soru üretmesi istendi.
5. Aktif adapter'dan aday adapter oluşturuldu ve tek soru-cevap çiftiyle 3 epoch eğitildi.
6. Benzer sorular üzerinde genel değerlendirme yapıldı.
7. Finalde tüm modeller bağımsız final test kümesinde karşılaştırıldı.

5. adımdan sonra akış iki stratejiye ayrılır:

**Selective kararı:** Benzer soru doğruysa aday adapter kabul edilir. Yanlışsa güncelleme geri alınır.  
**Blind kararı:** Benzer soru sonucu kaydedilir. Doğru ya da yanlış olsa da güncelleme tutulur.

İki yol da önce benzer sorular üzerinde genel değerlendirmeye, ardından bağımsız final test karşılaştırmasına bağlanır.

**Konuşmacı notu:**  
Deneyin temel yapısı önceki akıştaki gibi tek örnek üzerinden öğrenme ve hemen ardından benzer soru ile kontrol etme mantığına dayanıyor. Fark şu noktada ortaya çıkıyor: aynı LoRA eğitim adımından sonra Selective strateji benzer soru sonucuna göre güncellemeyi kabul ediyor ya da geri alıyor. Blind strateji ise aynı güncellemeyi herhangi bir geri alma yapmadan tutuyor.

---

## Slayt 5 - Benzer Soru Başarısı

![Benzer soru kümülatif doğruluk eğrisi](results/figures/similar_learning_curve.png)

- Selective: **405 / 500 doğru**, doğruluk **0.810**
- Blind: **389 / 500 doğru**, doğruluk **0.778**
- Benzer soru değerlendirmesinde Selective daha başarılıdır.

**Konuşmacı notu:**  
Benzer soru ara değerlendirmesinde beklenen sonuç büyük ölçüde gerçekleşti. Selective strateji, yalnızca başarılı görünen güncellemeleri tuttuğu için benzer sorular üzerinde Blind stratejiden daha yüksek doğruluk verdi. Bu sonuç yerel aktarım açısından Selective yaklaşımının daha iyi çalıştığını gösteriyor.

---

## Slayt 6 - Bağımsız Test Başarısı

![Final test doğruluğu](results/figures/test_accuracy.png)

| Model | Doğruluk Oranı | Doğruluk |
| --- | --- | --- |
| M1 baseline | 290 / 500 | 0.580 |
| Selective | 274 / 500 | 0.548 |
| Blind | 296 / 500 | 0.592 |

**Konuşmacı notu:**  
Asıl ilginç sonuç final testte ortaya çıktı. Benzer soru değerlendirmesinde daha iyi olan Selective strateji, bağımsız test kümesinde temel modelin altına düştü. Blind strateji ise benzer soru başarısında daha düşük görünmesine rağmen final testte en iyi sonucu verdi. Bu durum, tek bir benzer soruya göre verilen kabul kararının genel başarıyı garanti etmediğini gösteriyor.

---

## Slayt 7 - Düzeltme ve Bozma Dengesi

![Düzeltme ve bozma dengesi](results/figures/test_transitions.png)

- Selective, M1'in yanlış yaptığı **79** soruyu düzeltti.
- Selective, M1'in doğru yaptığı **95** soruyu bozdu.
- Blind, M1'in yanlış yaptığı **85** soruyu düzeltti.
- Blind, M1'in doğru yaptığı **79** soruyu bozdu.
- Net etki: Selective **-16**, Blind **+6**

**Konuşmacı notu:**  
Bu grafik final test sonucunun neden böyle çıktığını daha net gösteriyor. Her iki strateji de bazı eski hataları düzeltti, fakat bazı eski doğruları da bozdu. Selective stratejide bozulan doğru sayısı düzeltilen hata sayısından fazla olduğu için net etki negatif oldu. Blind stratejide ise düzeltilen hata sayısı biraz daha yüksek ve bozulan doğru sayısı daha düşük kaldı.

---

## Slayt 8 - Ana Yorum ve Sonuç

- Öğretmen çözümüyle yapılan tek örneklik LoRA güncellemesi, benzer sorulara aktarım sağlayabiliyor.
- Ancak benzer soru başarısı, bağımsız test başarısını garanti etmiyor.
- Selective yerel başarıda daha iyi, fakat final testte daha zayıf kaldı.
- Blind daha basit olmasına rağmen final testte daha iyi genelledi.
- Güncellemeler bazı hataları düzeltirken bazı eski doğruları bozabiliyor.

### Gelecek Çalışmalar

- Gelecek çalışmalarda genel performansı koruyacak güncelleme kabul kuralları geliştirilebilir.
- Daha çok veri seti ile ve farklı çözümler ile eğitim yapılabilir.
- Sadece sayısal cevap değil de çözümün tamamına odaklanan değerlendirme metrikleri kullanılabilir.

**Konuşmacı notu:**  
Çalışmanın temel sonucu şu: Bir güncellemenin hemen ardından gelen benzer soruda iyi görünmesi, onun genel test dağılımında da faydalı olacağı anlamına gelmiyor. Bu nedenle ileride tek benzer soru yerine çoklu doğrulama, validation tabanlı kabul, daha düşük öğrenme oranı veya replay gibi yöntemlerle daha güvenilir güncelleme stratejileri denenebilir.

---

## Kapanışta Söylenebilecek Kısa Özet

Bu çalışmada Qwen/Qwen3.5-4B modelinin GSM8K_TR üzerindeki hatalarından öğretmen model yardımıyla öğrenmesini inceledim. Selective strateji benzer sorularda daha iyi sonuç verdi, fakat bağımsız testte Blind strateji daha başarılı oldu. Bu bulgu, yerel benzer soru başarısının genel genelleme için yeterli bir sinyal olmadığını gösterdi.

---

## Sunuma Çevirirken Kullanılacak Görseller

- `results/figures/similar_learning_curve.png`
- `results/figures/test_accuracy.png`
- `results/figures/test_transitions.png`

## Sunum Süresi Önerisi

- Slayt 1: 30 saniye
- Slayt 2: 60 saniye
- Slayt 3: 60 saniye
- Slayt 4: 60 saniye
- Slayt 5: 75 saniye
- Slayt 6: 75 saniye
- Slayt 7: 75 saniye
- Slayt 8: 60 saniye

Toplam yaklaşık süre: 7-8 dakika
