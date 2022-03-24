import torch

from einops import rearrange, repeat

def make_mask(tensor: torch.tensor, option: str) -> torch.Tensor:
    '''
    tensor (batch_size, seq_len)
    '''
    if option == 'padding':
        tmp = torch.full_like(tensor, fill_value=0)
        mask = (tensor != tmp).float()
        mask = rearrange(mask, 'bs seq_len -> bs 1 1 seq_len')
    elif option == 'lookahead':
        padding_mask = make_mask(tensor, 'padding')
        padding_mask = repeat(padding_mask, 'bs 1 1 k_len -> bs 1 new k_len', new=padding_mask.shape[3])

        mask = torch.ones_like(padding_mask)
        mask = torch.tril(mask)

        mask = mask * padding_mask
    
    return mask

if __name__ == "__main__":


    test = torch.Tensor([[1,2,3,4,5,6]])
    print(test.shape)
    test1 = make_mask(test, option='padding')
    test2 = make_mask(test, option='lookahead')
    
    print(test1.shape)
    print(test1)
    print(test2.shape)
    print(test2)