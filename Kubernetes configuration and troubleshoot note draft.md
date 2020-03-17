# Kubernetes configuration and troubleshoot note (draft)

Created By: [@guzhaqixin](https://github.com/guzhaqixin)

This doc records the [Kubernetes](https://github.com/kubernetes/kubernetes) configuration and troubleshoot process for this [project](https://github.com/MrZhang1994/crowdsensing). It is largely affected by the specific appliances used and corresponding history environment, thus YMMV. The doc by present is maintained by [@gzqx](https://github.com/gzqx) and please refer to him if there is any problem.

# Environment setup

## Appliance

Two x86 computer with Ubuntu 18.04. The computer which contains master node will be named as A and slaves named as B, C and D...

## Docker

Docker comes with OS installation is ce version and reinstall it to docker.io.

A problem occurred in A as

    /proc/sys/net/bridge/bridge-nf-call-iptables does not exist

which can be solved through

    modprobe br_netfilter

## Kubernetes Components

Installed according to [official document](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/).

Originally flannel was planned to be used as the network pod but [problem](https://github.com/kubernetes/kubernetes/issues/86587) occurs.

Then changed to use calico. 

### Calico Configuration

The latest version of Calico manifest file has an unsolved bug. The auto ip detection may fails when interface configuration of host of different nodes are different. Presently we specify the certain network interface name we use of each host computer. Notice that such configuration may cause problem when name of network interface changes (which may be caused by VPN usage etc.)

For detail one may refer to this [issue](https://github.com/projectcalico/calico/issues/2561).
