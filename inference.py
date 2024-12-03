# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import os.path as op

from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.simple_tokenizer import SimpleTokenizer
from utils.iotools import load_train_configs
import torch.nn.functional as F
from datasets.cuhkpedes import CUHKPEDES
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image



def main():
    # config
    config_file = "logs/CUHK-PEDES/irra_cuhk/configs.yaml"
    args = load_train_configs(config_file)

    args.training = False
    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, "best.pth"))
    model.to(device)
    
    model.eval()

    def compute_embedding(model, txt_loader, img_loader):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids

       
    def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            if truncate:
                tokens = tokens[:text_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {caption} is too long for context length {text_length}"
                )
        result[:len(tokens)] = torch.tensor(tokens)
        return result
    
    
    def infer(caption, model):
        model = model.eval()
        device = next(model.parameters()).device
        gids, gfeats = [], []
        for pid, img in test_img_loader:
                img = img.to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(img)
                gids.append(pid.view(-1)) # flatten 
                gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        
        
        # text
        text_length =  77
        tokenizer = SimpleTokenizer()
        caption = tokenize(caption, tokenizer=tokenizer, text_length=text_length, truncate=True)
        caption = Variable(caption).unsqueeze(0)
        caption = caption.to(device)

        with torch.no_grad():
            text_feat = model.encode_text(caption)
        similarity = text_feat @ gfeats.t()
        _, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)  # q * topk
        indices = indices.cpu()
        
        idx = 0
        dataset = CUHKPEDES(root='./data')
        test_dataset = dataset.test
        
        a = indices[idx]
        print(a)
        print(gids[a])
       
        image_paths=[test_dataset['img_paths'][i] for i in a] 
        print(image_paths)
        

        col = len(a)
        plt.subplot(1, col+1, 1)
        plt.xticks([])
        plt.yticks([])
        for i in range(col):
            plt.subplot(1, col+1, i+1)
            img = Image.open(image_paths[i])
            img = img.resize((128, 256))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
       
        plt.savefig("result-img", dpi=300)
    
    
    
    caption = "small height and blue pant"
    infer(caption, model)

if __name__ == "__main__":
    main()
