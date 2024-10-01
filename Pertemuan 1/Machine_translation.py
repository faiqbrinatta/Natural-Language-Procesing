# from transformers import MarianMTModel, MarianTokenizer

# nama_model = 'Helsinki-NLP/opus-mt-jv-en' # model jawa ke inggris
# tokenizer = MarianTokenizer.from_pretrained(nama_model)
# model = MarianMTModel.from_pretrained(nama_model)

# # fungsi untuk penerjemahan

# def translate(text,model,tokenizer):
#     translated = model.generate(**tokenizer(text,return_tensors="pt", padding=True))
#     return[tokenizer.decode(t, skip_special_tokens=True)for t in translated]

# kalimat_jawa = "sugeng rawuh"
# hasil_terjemahan = translate(kalimat_jawa,model,tokenizer)
# print(hasil_terjemahan[0])



from transformers import MarianMTModel, MarianTokenizer

# Memilih model yang tepat
nama_model = 'Helsinki-NLP/opus-mt-jv-en'  # Model untuk terjemahan Jawa ke Inggris
tokenizer = MarianTokenizer.from_pretrained(nama_model)
model = MarianMTModel.from_pretrained(nama_model)

# Fungsi untuk melakukan terjemahan
def translate(text, model, tokenizer):
    # Tokenisasi input
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Generasi terjemahan menggunakan model
    translated = model.generate(**inputs)
    # Decode hasil terjemahan ke bentuk teks
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Kalimat bahasa Jawa yang akan diterjemahkan
kalimat_jawa = "sugeng rawuh"
hasil_terjemahan = translate(kalimat_jawa, model, tokenizer)

# Menampilkan hasil terjemahan
print(hasil_terjemahan[0])
