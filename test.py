import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import os

# project imports.
from utils.options import get_args
from loaders import create_loader
from networks import create_network
from utils.boundary_metric import compute_bfscore_batch_with_load
from utils.evaluation import Evaluator


def evaluate_bf_score_with_image_store(args):
    device = torch.device(args.device)
    # Init dataset and data loader
    _, test_loader, n_classes = create_loader(args.dataset, args.batch_size)
    num_cpu_workers = args.num_threads
    # # Initialize the networks.
    network = create_network(args, n_classes)
    network = network.to(device)
    # Load up the parameters of the networks.
    load_name = f'{args.dataset}-{network.name}'
    network.load_model(load_name, device)
    # # Record which inputs should be used.
    use_depth = args.use_depth
    # # Then do the loop over the test set.
    output_dir = os.path.join(os.getcwd(), 'output', 'bf_temp')
    os.mkdir(output_dir)
    save_idx = 0
    gt_paths, pred_paths = [], []
    with torch.no_grad():
        for idx, (image, depth, semantic) in tqdm(enumerate(test_loader)):
            # Select what the networks should receive as input and then move that to GPU.
            image, depth, semantic = image, depth, semantic.to(device)
            if use_depth:
                prediction = network(depth.to(device))
            else:
                prediction = network(image.to(device))
            # Get the max prediction, which is the integer of the class.
            _, pred_label = torch.max(prediction, dim=1)
            gts = [(semantic[i, 0].cpu().numpy() + 1).astype(np.uint8) for i in range(prediction.shape[0])]
            preds = [(pred_label[i].cpu().numpy() + 1).astype(np.uint8) for i in range(prediction.shape[0])]
            for idx in range(len(gts)):
                gt_path = os.path.join(output_dir, f'gt_{save_idx:05}.npy')
                np.save(gt_path, gts[idx])
                gt_paths.append(gt_path)
                pred_path = os.path.join(output_dir, f'pred_{save_idx:05}.npy')
                np.save(pred_path, preds[idx])
                pred_paths.append(pred_path)
                save_idx += 1
    torch.cuda.empty_cache()
    # Now make a job array.
    jobs = [(gt, pred) for gt, pred in zip(gt_paths, pred_paths)]
    bf_scores = Pool(num_cpu_workers).map(compute_bfscore_batch_with_load, jobs)
    test_set_bfscore = sum(bf_scores) / float(len(bf_scores))
    r_str = f'{args.dataset}-{network.name} BF: {test_set_bfscore:.4f}\n'
    modality = "D" if network.n_inputs == 1 else "I"
    with open(f'result_{args.dataset}-{modality}', 'a') as result_file:
        result_file.write(r_str)
    print(r_str)
    os.rmdir(output_dir)


def evaluate(args):
    device = torch.device(args.device)
    # Init dataset and data loader
    _, test_loader, n_classes = create_loader(args.dataset, args.batch_size)
    # Initialize the networks.
    network = create_network(args, n_classes)
    network = network.to(device)
    # Load up the parameters of the networks.
    load_name = f'{args.dataset}-{network.name}'
    network.load_model(load_name, device)
    network.eval()
    # Do evaluation.
    evaluator = Evaluator(network, test_loader, device, args.use_depth)
    miou, fw_miou, p_acc, c_acc = evaluator.evaluate()
    modality = "D" if network.n_inputs == 1 else "I"
    r_str = f'{args.dataset}-{network.name}' \
            f'\tmIoU: {miou:.4f} fw_mIoU: {fw_miou:.4f} |' \
            f' P_ACC: {p_acc:.4f} C_ACC: {c_acc:.4f}\n'
    with open(f'result_{args.dataset}-{modality}', 'a') as result_file:
        result_file.write(r_str)
    print(r_str)


if __name__ == '__main__':
    args = get_args()
    print("Running code using CUDA {}".format(torch.version.cuda))
    evaluate(args)
    torch.cuda.empty_cache()
    evaluate_bf_score_with_image_store(args)

    print("YOU ARE TERMINATED!")
