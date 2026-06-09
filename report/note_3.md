Faz 6 test değerlendirmesi tamamlandı.

Ana sonuç:

| Model | Doğru / 500 | Accuracy | M1'in yanlış yaptığı 210 test sorusunda doğru | M1'e göre net fark |
|-------|-------------|----------|----------------------------------------------|--------------------|
| M1 baseline | 290 | 0.580 | 0 | 0 |
| Selective final adapter | 274 | 0.548 | 79 | -16 |
| Blind final adapter | 296 | 0.592 | 85 | +6 |

Selective final adapter:

- Test accuracy: 274 / 500 = 0.548
- M1'in yanlış yaptığı 210 test sorusundan 79'unu doğru çözdü.
- M1'in doğru yaptığı 290 test sorusundan 195'ini korudu, 95'ini bozdu.
- Net etki: +79 düzeltilen eski hata, -95 bozulan eski doğru = -16.

Blind final adapter:

- Test accuracy: 296 / 500 = 0.592
- M1'in yanlış yaptığı 210 test sorusundan 85'ini doğru çözdü.
- M1'in doğru yaptığı 290 test sorusundan 211'ini korudu, 79'unu bozdu.
- Net etki: +85 düzeltilen eski hata, -79 bozulan eski doğru = +6.

Selective ve blind karşılaştırması:

- İkisi de doğru: 217
- Sadece selective doğru: 57
- Sadece blind doğru: 79
- İkisi de yanlış: 147

Yorum:

Faz 5 sonunda selective strateji benzer soru değerlendirmesinde blind stratejiden daha iyi görünüyordu: selective 405/500, blind 389/500. Faz 6 test değerlendirmesi ise bu lokal üstünlüğün sabit test kümesine taşınmadığını gösterdi.

Selective strateji kötü görünen bazı güncellemeleri geri aldığı için benzer soru testinde daha kontrollüydü. Ancak final testte M1'in doğru yaptığı fazla sayıda soruyu bozdu ve baseline'ın altına indi. Blind strateji benzer soru testinde daha zayıf görünmesine rağmen, tüm güncellemeleri tuttuğu için final testte az da olsa daha iyi genelledi.

Bu sonuç şu yorumu destekliyor:

Bir sorunun teacher çözümünü öğrenmek bazı benzer ve test sorularında fayda sağlayabiliyor; ancak tek adımlık benzer soru başarısı, güncellemenin uzun vadeli test genellemesini güvenilir biçimde tahmin etmiyor. Bu deneyde kör strateji final test accuracy bakımından en iyi sonucu verdi.

Çıktı dosyaları:

- `logs/selective_test_summary.json`
- `logs/blind_test_summary.json`
- `data/test_selective_predictions.jsonl`
- `data/test_blind_predictions.jsonl`
- `results/evaluation.csv`
