"""CPU fallback test for native tensors and unregistered deployment shapes."""
import torch
from full_runtime import CompleteProjection


def cpu():
    generator=torch.Generator().manual_seed(435389)
    layer=torch.nn.Linear(64,32,bias=False,dtype=torch.bfloat16).eval()
    with torch.no_grad():
        layer.weight.normal_(generator=generator)
        runner=CompleteProjection(layer)
        for shape in ((1,64),(19,64),(53,64)):
            x=torch.randn(*shape,generator=generator,dtype=torch.bfloat16)
            assert torch.equal(runner(x).view(torch.int16),layer(x).view(torch.int16))
            assert runner.prepare([x],[x.clone()])=={"reason":"unsupported or inconsistent geometry"}
        runner.close()
    print("unknown CPU layout, native fallback, changing lengths and bits passed")


if __name__=="__main__": cpu()

