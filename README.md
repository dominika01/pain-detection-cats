# An AI model for detecting pain in cats

This project uses AI to detect signs of pain in cats based on the Feline Grimace Scale.

To use it, run the main.py file with the path to the image you want to analyse as a command line argument:

```
python3 main.py path/to/your/image
```

Similarly to how a vet would do it, the program splits the image into features, rates each of them on a 0-2 scale, and gives an overall score.

This is an example output:

> Results
> <br><br>
> 
> Ear position&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;2
>
> Orbital tightening&emsp;&emsp;&emsp;&emsp;&ensp;1
>
> Muzzle tension&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;2
>
> Whiskers position&emsp;&emsp;&emsp;&emsp;&nbsp;0
>
> Head position&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;1
> <br><br>
>
> Overall impression&emsp;&emsp;&emsp;&emsp;1


Legend:

0 - action unit absent

1 - moderately present

2 - markedly present
