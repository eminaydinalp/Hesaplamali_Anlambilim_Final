Kısa yorumum: seçici strateji ara ölçümde kör stratejiden daha iyi görünüyor, ama nihai karar için Faz 6 test değerlendirmesi şart.

Ara sonuç:

Selective loop:
500 adım
405 doğru / kabul
95 yanlış / geri alma
Benzer soru başarısı: 405 / 500 = %81.0

Blind loop:
500 adım
389 doğru
111 yanlış
Tüm güncellemeler tutuldu
Benzer soru başarısı: 389 / 500 = %77.8
Yani seçici strateji, benzer soru değerlendirmesinde kör stratejiden:

+16 doğru örnek
+3.2 puan
önde.

Adım adım karşılaştırma da şöyle:

İkisi de doğru:        359
Sadece selective doğru: 46
Sadece blind doğru:     30
İkisi de yanlış:        65
Bu güzel bir sonuç çünkü selective loop’un yaptığı şeyin anlamlı olduğunu gösteriyor: bazı kötü güncellemeleri geri aldığı için blind’a göre 46 örnekte avantaj sağlamış. Ama blind’ın da 30 örnekte selective’den iyi olması önemli; bu, bazı “o anda yanlış görünen” güncellemelerin ileride faydalı olabileceğini gösteriyor olabilir.

Dikkat edilmesi gereken nokta:

blind accepted_rows = 500
bu “500 doğru yaptı” demek değil. Blind stratejide her güncelleme tutulduğu için accepted_rows doğal olarak 500. Gerçek ara başarı metriği:

similar_correct_rows = 389
Bence şu ana kadarki yorum:

Seçici strateji, lokal benzer soru testinde kör stratejiden daha kontrollü ve biraz daha başarılı. Ama proje sorusunun asıl cevabı final test setinde çıkacak: selective final adapter mı, blind final adapter mı, yoksa base Qwen mı testte daha iyi genelliyor?