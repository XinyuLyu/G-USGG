from maskrcnn_benchmark.clip import clip

subj = 'man'
rel = 'holding'
obj = 'plate'
sentense = "A photo of a " + subj + ' ' + rel +' '+ obj
clip_model, _ = clip.load("ViT-B/32", device='cpu')
triplet_tokens = clip.tokenize(sentense)
triplet_tensor = clip_model.encode_text(triplet_tokens).cuda()
print()


