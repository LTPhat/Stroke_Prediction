import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import base64
import Preprocessing
import pandas as pd
import numpy as np
import warnings
from PIL import Image
warnings.filterwarnings('ignore')
model = pickle.load(open('model.pkl', 'rb'))
test_data = Preprocessing.X_test_copy

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('E:\StrokeProject\Main_Project\Back9.png')
def create_data_input(age,gender,BMI,work_type,mean_blood_pressure,cholesterol,smoke,avg_glu_level,alcohol,active,data = test_data):
    d = {'age': float(age),'gender':gender,
         'BMI':float(BMI),'work_type':work_type,
         'mean_blood_pressure':float(mean_blood_pressure),
         'cholesterol':cholesterol,'smoke':int(smoke),
         'avg_glu_level':float(avg_glu_level),
         'alcohol':int(alcohol),'active':int(active)}
    datapoint = pd.DataFrame(data = d,index = [len(test_data)])
    return datapoint
def add_to_test(data1,data2): #data 1 is new point, #data 2 is test_data
    data2.reset_index(drop=True)
    data_after = pd.concat((data2,data1),axis=0)
    return data_after

def prediction(data):
    data= np.array(data).reshape(1,-1)
    predict = model.predict_proba(data)
    return predict[0][1]
def main():
    st.markdown("<h1 style='text-align:center; color: black;'>Mô hình dự đoán tỉ lệ đột quỵ</h1>",
                unsafe_allow_html=True)
    html_temp = """
    <div style="background-color: powderblue ;padding:15px">
    <h3 style="color:black;text-align:center; font-size: 15 px"> Hãy nhập những thông số cá nhân để có kết
    quả dự đoán</h3>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("")
    html_temp2 = """
    <h5 style="color:black; font-size: 5 px"> Chú ý: Những ô yêu cầu điền chữ thì điền đúng từ khóa đã được chỉ định. </h5>
    """
    st.markdown(html_temp2, unsafe_allow_html=True)
    name = st.text_input('Nhập họ và tên:',"Type here")
    age = st.text_input("Tuổi","Type here")
    gender = st.text_input("Giới tính ","Type Male/Female")
    BMI = st.text_input("Chỉ số BMI","Type here")
    st.write(
        "Truy cập vào [đây](https://www.calculator.net/bmi-calculator.html?ctype=metric&cage=25&csex=m&cheightfeet=5&cheightinch=10&cpound=160&cheightmeter=160&ckg=120&printit=0&x=47&y=22) để tính chỉ số BMI.")
    work_type = st.text_input("Công việc (Govt_job: nhà nước; Private: tư nhân; Self-employed: tự quản lí; Other: khác)","Gõ Govt_job, Private,...")
    mean_blood_pressure = st.text_input("Chỉ số huyết áp trung bình (Ví dụ: 135/86 mmHg thì thực hiện (135+86*2)/3 rồi điền kết quả)","Type here")
    cholesterol = st.text_input("Mức độ cholesterol (Normal: bình thường; Above Normal: trên mức bình thường; High: cao)","Type here")
    smoke = st.text_input("Mức độ sử dụng thuốc lá (Thường xuyên: 1; Ít hoặc không có: 0)","Type 0 or 1")
    avg_glu_level = st.text_input("Chỉ số đường huyết trung bình (mg/dL)","Type here")
    st.write("Truy cập vào [đây](https://dankhang.vn/wp-content/uploads/2022/03/chi-so-HbA1c.jpg) để biết thêm thông tin về chỉ số đường huyết trung bình.")
    alcohol = st.text_input("Mức độ sử dụng đồ uống có cồn (Thường xuyên: 1; Ít hoặc không có: 0)","Type 0 or 1")
    active = st.text_input("Mức độ vận động (Thường xuyên: 1; Ít hoặc không có: 0)","Type 0 or 1")
    if st.button("Dự đoán"):
        new_point = create_data_input(age, gender, BMI, work_type, mean_blood_pressure, cholesterol, smoke, avg_glu_level,
                              alcohol, active, data=test_data)
        data_final = add_to_test(new_point,test_data)
        # Categorical encode
        df_dummy = data_final[['gender', 'work_type', 'cholesterol']]
        df_dummy = pd.get_dummies(df_dummy)
        data_temp = data_final.drop(['gender', 'work_type', 'cholesterol'], axis=1)
        data_final = pd.concat([data_temp, df_dummy], axis=1)
        data_final = data_final.reset_index(drop=True)
        #Scaler
        scaler = StandardScaler()
        numeric_col = ['age', 'BMI', 'mean_blood_pressure', 'avg_glu_level']
        data_final[numeric_col] = scaler.fit_transform(data_final[numeric_col])
        output = prediction(data_final.iloc[len(data_final)-1,:])



        if output >= 0.6:
            st.error('Xác suất mắc đột quy dự đoán là: {}{}'.format(np.round(output*100,2),'%'))
            image1 = Image.open('error.png')
            st.image(image1)
            html_temp3 = """
            <div style="background-color: blanchedalmond ;padding:15px">
    <h3 style="color:navy;text-align:center; font-size: 15 px"> Hãy cảnh giác! Bạn thuộc nhóm người được dự đoán có tỉ lệ mắc đột quỵ cao</h3>
    </div>
            """
            st.markdown(html_temp3,unsafe_allow_html=True)
            st.markdown("")
            html_temp4 = """
                <div style="background-color:aqua;padding:5px;text-align: center">
                <h4 style="color:navy; font-size: 10 px"> Lời khuyên: </h4>
                </div>
                """
            st.markdown(html_temp4, unsafe_allow_html=True)
            st.markdown("")
            html_temp_1 = """
            <h5 style="color:navy; font-size: 5 px"> 1) Giữ huyết áp ở mức lý tưởng. </h5>
            """
            st.markdown(html_temp_1, unsafe_allow_html=True)
            html_temp_2 = """
            <h5 style="color:navy; font-size: 5 px"> 2) Giữ chỉ số BMI thấp hơn 25. </h5>
            """
            st.markdown(html_temp_2, unsafe_allow_html=True)
            html_temp_3 = """
            <h5 style="color:navy; font-size: 5 px"> 3) Thường xuyên tập thể dục. </h5>
            """
            st.markdown(html_temp_3, unsafe_allow_html=True)
            html_temp_4 = """
                        <h5 style="color:navy; font-size: 5 px"> 4) Hạn chế thức uống có cồn. </h5>
                        """
            st.markdown(html_temp_4, unsafe_allow_html=True)
            html_temp_5 = """
                                    <h5 style="color:navy; font-size: 5 px"> 5) Không hút thuốc lá. </h5>
                                    """
            st.markdown(html_temp_5, unsafe_allow_html=True)





        elif (output >= 0.3) and (output < 0.6):
            st.warning('Xác suất mắc đột quy dự đoán là: {}{}'.format(np.round(output*100,2),'%'))
            image2 = Image.open('attention.jpg')
            st.image(image2)
            html_temp_out1 = """
                        <div style="background-color: blanchedalmond ;padding:15px">
                <h3 style="color:navy;text-align:center; font-size: 15 px"> Chú ý! Bạn thuộc nhóm người được dự đoán có tỉ lệ mắc đột quỵ ở mức trung bình</h3>
                </div>
                        """
            st.markdown(html_temp_out1, unsafe_allow_html=True)
            st.markdown("")
            html_temp4 = """
                            <div style="background-color:aqua;padding:5px;text-align: center">
                            <h4 style="color:navy; font-size: 10 px"> Lời khuyên: </h4>
                            </div>
                            """
            st.markdown(html_temp4, unsafe_allow_html=True)
            st.markdown("")
            html_temp_1 = """
                        <h5 style="color:navy; font-size: 5 px"> 1) Giữ huyết áp ở mức lý tưởng. </h5>
                        """
            st.markdown(html_temp_1, unsafe_allow_html=True)
            html_temp_2 = """
                        <h5 style="color:navy; font-size: 5 px"> 2) Giữ chỉ số BMI thấp hơn 25. </h5>
                        """
            st.markdown(html_temp_2, unsafe_allow_html=True)
            html_temp_3 = """
                        <h5 style="color:navy; font-size: 5 px"> 3) Thường xuyên tập thể dục. </h5>
                        """
            st.markdown(html_temp_3, unsafe_allow_html=True)
            html_temp_4 = """
                                    <h5 style="color:navy; font-size: 5 px"> 4) Hạn chế thức uống có cồn. </h5>
                                    """
            st.markdown(html_temp_4, unsafe_allow_html=True)
            html_temp_5 = """
                                                <h5 style="color:navy; font-size: 5 px"> 5) Không hút thuốc lá. </h5>
                                                """
            st.markdown(html_temp_5, unsafe_allow_html=True)




        else:
            st.success('Xác suất mắc đột quỵ dự đoán là: {}{}'.format(np.round(output * 100, 2), '%'))
            image3 = Image.open('safe.png')
            st.image(image3)
            html_temp_out_3 = """
                                    <div style="background-color: green ;padding:15px">
                            <h3 style="color:navy;text-align:center; font-size: 15 px">Bạn thuộc nhóm người được dự đoán có tỉ lệ mắc đột quỵ ở mức thấp</h3>
                            </div>
                                    """
            st.markdown(html_temp_out_3, unsafe_allow_html=True)
    # Lưu input của người dùng vào file excel:
        # Load excel_file:
        list_data = pd.read_excel('E:\StrokeProject\Main_Project\input_user.xlsx',index_col=0)
        list_data.reset_index()
        # Create input datapoint
        d = {'name': name,'age': float(age),'gender':gender,
         'BMI':float(BMI),'work_type':work_type,
         'mean_blood_pressure':float(mean_blood_pressure),
         'cholesterol':cholesterol,'smoke':int(smoke),
         'avg_glu_level':float(avg_glu_level),
         'alcohol':int(alcohol),'active':int(active),'stroke_prob':np.round(output,2)}
        new_input = pd.DataFrame(data = d,index = [len(list_data)])
        list_data = pd.concat((list_data,new_input),axis=0)
        list_data.to_excel('E:\StrokeProject\Main_Project\input_user.xlsx')
if __name__=='__main__':
    main()
