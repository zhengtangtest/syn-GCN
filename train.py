from data_utils import *
from score import score
import os

def train(training_data, encoder, classifier, encoder_optimizer, classifier_optimizer, criterion):

    encoder.train()
    classifier.train()

    for datapoint in training_data:
        
        relation    = torch.tensor([datapoint[0]], dtype=torch.long, device=device)
        sentence    = tensorFromIndexes(datapoint[1])
        edge_index  = torch.tensor(datapoint[2], dtype=torch.long, device=device)
        label_index = tensorFromIndexes(datapoint[3])
        entity_subj = datapoint[4]
        entity_obj  = datapoint[5]
        type_subj   = datapoint[6]
        type_obj    = datapoint[7]

        outputs, subj, obj, sw, ow, syn_embedded = encoder(sentence, label_index,
            entity_subj, entity_obj, edge_index)

        predict = classifier(outputs, syn_embedded, subj, obj, edge_index)

        loss = criterion(predict, relation)

        loss.backward()

        clipping_value = 1#arbitrary number of your choosing
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)

        encoder_optimizer.step()
        classifier_optimizer.step()

def eval(dev_data, encoder, classifier):

    encoder.eval()
    classifier.eval()
    
    key        = list()
    prediction = list()

    with torch.no_grad():

        for datapoint in dev_data:
            
            relation    = id2relation[datapoint[0]]
            sentence    = tensorFromIndexes(datapoint[1])
            edge_index  = torch.tensor(datapoint[2], dtype=torch.long, device=device)
            label_index = tensorFromIndexes(datapoint[3])
            entity_subj = datapoint[4]
            entity_obj  = datapoint[5]
            type_subj   = datapoint[6]
            type_obj    = datapoint[7]

            key.append(relation)

            outputs, subj, obj, sw, ow, syn_embedded = encoder(sentence, label_index,
                entity_subj, entity_obj, edge_index)

            predict = classifier(outputs, syn_embedded, subj, obj, edge_index)

            topv, topi = predict.topk(1)

            prediction.append(id2relation[topi.item()])

    score(key, prediction, verbose=True)



if __name__ == '__main__':

    input_lang, dep_lang, training_data = load_data('tacred/data/json/train.json')

    dev_data = load_data('tacred/data/json/dev.json', input_lang, dep_lang)

    embeds, embedding_size = load_embeddings("glove.840B.300d.txt", input_lang)
    embeds = torch.FloatTensor(embeds)
    hidden_size = 100
    learning_rate = 0.00005
    criterion = nn.NLLLoss()

    encoder    = EncoderRNN(input_lang.n_words, embedding_size, 
        dep_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(hidden_size, hidden_size, len(id2relation)).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(100):
        random.shuffle(training_data)
        train(training_data, encoder, classifier, encoder_optimizer, classifier_optimizer, criterion)

        eval(dev_data, encoder, classifier)

        os.mkdir("model/%d"%epoch)
        PATH = "model/%d"%epoch
        torch.save(encoder, PATH+"/encoder")
        torch.save(classifier, PATH+"/classifier")