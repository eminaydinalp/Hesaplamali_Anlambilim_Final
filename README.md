# GSM8K_TR Seçici Fine-Tuning Deneyi

Bu proje, GSM8K_TR üzerinde bir modelin çözemediği soruların öğretmen model çözümleriyle fine-tune edilmesinin benzer sorulara genelleme sağlayıp sağlamadığını incelemek için hazırlanmıştır.

Ana karşılaştırma iki strateji arasındadır:

- Seçici strateji: Model, yeni öğrendiği çözümle benzer soruyu çözebiliyorsa güncellenmiş modelle devam eder.
- Kör strateji: Her fine-tuning adımından sonra son modelle devam eder.

## Klasör Yapısı

```text
.
├── data/
├── models/
├── results/
├── scripts/
└── logs/
```

## Ortam Kurulumu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cihaz Kontrolü

CUDA, MPS veya CPU fallback durumunu kontrol etmek için:

```bash
source .venv/bin/activate
python scripts/check_device.py
```

Script çıktıyı ekrana basar ve aynı bilgiyi `logs/device_check.json` dosyasına yazar.

## Proje Fazları

Ayrıntılı iş takibi `Progression.md` dosyasındadır. Faz 0 ortam kurulumu ve temel klasör yapısını kapsar.
