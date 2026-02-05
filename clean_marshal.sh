find sgm -name "*.pyc" -delete
find sgm -name "__pycache__" -type d -exec rm -r {} + 
rm -rf __pycache__