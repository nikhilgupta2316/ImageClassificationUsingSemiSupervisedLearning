models = ["softmax", "twolayernn", "threelayernn", "onelayercnn", "twolayercnn", "vggnet", "alexnet", "resnet"]
settings = ["", "-4k", "-ssl"]
for model in models:
    for setting in settings:
        job_name = model + setting
        text = "#!/bin/bash"
        text += "\n"
        text += "#SBATCH --job-name=" + job_name
        text += "\n"
        text += "#SBATCH --output=sbatch_logs/run-%j-"+ job_name +".out"
        text += "\n"
        text += "#SBATCH --error=sbatch_logs/run-%j-"+ job_name +".err"
        text += "\n"
        text += "#SBATCH --gres gpu:1"
        text += "\n"
        text += "#SBATCH --nodes 1"
        text += "\n"
        text += "#SBATCH --ntasks-per-node 1"
        text += "\n"
        text += "#SBATCH --partition=short"
        text += "\n"
        text += "\n"
        text += "set -x"
        text += "\n"
        text += "sacct -j ${SLURM_JOB_ID} --format=User%20,JobID,Jobname%40,partition,state,time,start,nodelist"
        text += "\n"
        text += "bash scripts/run_" + job_name + ".sh"
        filename = job_name + ".sh"
        with open(filename, "w") as text_file:
            text_file.write(text)
