from .downup import Network, UpDownNet


def create_network(args, num_classes) -> Network:
    num_inputs = 1 if args.use_depth else 3
    return UpDownNet(num_inputs, num_classes, args.pool_method, args.unpool_method, args.conv_scheme,
                     pool_ks=args.pool_ks, unpool_ks=args.unpool_ks, conv_ks=args.conv_ks)
