# Text Classification
**Text classification** is the task of assigning a sentence or document an appropriate category. The categories depend on the chosen dataset and can range from topics.

## Models Detail:
- **fastText**:  
  Implmentation of <a href="https://arxiv.org/abs/1607.01759">Bag of Tricks for Efficient Text Classification</a> after embed each word in the sentence, this word representations are then averaged into a text representation, which is in turn fed to a linear classifier.it use softmax function to compute the probability distribution over the predefined classes. then cross entropy is used to compute loss. bag of word representation does not consider word order. in order to take account of word order, n-gram features is used to capture some partial information about the local word order; when the number of classes is large, computing the linear classifier is computational expensive. so it usehierarchical softmax to speed training process.
  - use bi-gram and/or tri-gram
  - use NCE loss to speed us softmax computation(not use hierarchy softmax as original paper)

  *Result*: performance is as good as paper, speed also very fast.

- TextCNN (*TODO*)
- Bert:Pre-training of Deep Bidirectional Transformers for Language Understanding (*TODO*)
- TextRNN (*TODO*)
- RCNN (*TODO*)
- Hierarchical Attention Network (*TODO*)  
- seq2seq with attention (*TODO*)
- Transformer("Attend Is All You Need") (*TODO*)
- Dynamic Memory Network (*TODO*)
- EntityNetwork:tracking state of the world (*TODO*)
- Ensemble models (*TODO*)
- Boosting (*TODO*)