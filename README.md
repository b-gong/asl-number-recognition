# asl-number-recognition
Testing and comparing various methods of gesture recognition

Commands:
- Baseline: python baseline/process_data.py
- CNN (enter cnn folder):
  - python cnn.py cnn
  - python cnn_complex.py
  - To test different models, replace the expression on line 419: model = ASLCNN_OneLayer()
    - ASLCNN_OneLayer()
    - ASLCNN_V0()
    - ASLCNN_V1()
    - ASLCNN_V2()
    - ASCLNN_V3()
    - ASCLNN_V4()
    - ASCLNN_V5()
    - ASVLNN_V6()
