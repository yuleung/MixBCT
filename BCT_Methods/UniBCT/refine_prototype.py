import torch
import numpy as np
import gc


def PrototypeGeneration(cfg, new_model, old_model, loader, rank, lamb=0.9, temper=0.05):
    with torch.no_grad():
        old_features = []
        new_features = []
        old_labels = []
        new_labels = []

        print(len(loader))
        if rank == 0:
            old_model.eval()
            old_model = old_model.to('cuda:0')
            for i, (img, label, _) in enumerate(loader):  # extract old and new features

                if i % 10000 == 0:
                    print('0: {}_{}'.format(i, i * cfg.batch_size))
                # print(img.dtype)
                old_local_embeddings = old_model(img)
                old_features.append(old_local_embeddings.detach().cpu().numpy())
                old_labels += label.detach().cpu().numpy().tolist()

            old_features = np.array(list(np.vstack(np.array(old_features[:-1]))) + list(np.array(old_features[-1])))

            print('empty')
            import time
            time.sleep(10)
            print('sleeped')

        if rank == 1:
            new_model.eval()
            for i, (img, label, _) in enumerate(loader):  # extract old and new features
                new_local_embeddings, _ = new_model(img)
                if i % 10000 == 0:
                    print('1: {}_{}'.format(i, i * 128))
                new_features.append(new_local_embeddings.detach().cpu().numpy())
            new_features = np.array(list(np.vstack(np.array(new_features[:-1]))) + list(np.array(new_features[-1])))
            np.save(cfg.output + '/tmp_new_features.npy', new_features)
            del new_features
            # gc.collect([new_features])

    torch.distributed.barrier()
    print('barriered!')
    if rank == 0:
        cls_num = len(set(old_labels))
        new_feat_list = [[] for _ in range(cls_num)]
        old_feat_list = [[] for _ in range(cls_num)]
        new_features = np.load(cfg.output + '/tmp_new_features.npy')

        old_prototype = torch.zeros(cls_num, new_features.shape[1])

        for i, label in enumerate(old_labels):  # aggregate by category
            new_feat_list[label].append(new_features[i, :])
            old_feat_list[label].append(old_features[i, :])

        for label in old_labels:
            old_vertices = torch.tensor(np.array(old_feat_list[label]))
            new_vertices = torch.tensor(np.array(new_feat_list[label]))
            edges = torch.mm(new_vertices, new_vertices.t()) / temper
            identity = torch.eye(edges.size(0))
            mask = torch.eye(edges.size(0), edges.size(0)).bool()
            edges.masked_fill_(mask, -1e9)
            edges = torch.softmax(edges, dim=0)
            # Eq. (9)
            edges = (1 - lamb) * torch.inverse(identity - lamb * edges)
            old_vertices = torch.mm(edges, old_vertices)
            # Eq. (10)
            # print(old_vertices.shape)
            old_prototype[label] = torch.mean(old_vertices, dim=0)
        np.save(cfg.output + '/old_prototype.npy', old_prototype)
        return