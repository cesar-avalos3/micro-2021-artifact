sudo echo "options nvidia \"NVreg_RestrictProfilingToAdminUsers=0\"" > /etc/modprobe.d/profile.conf
sudo update-initramfs -u
sudo reboot