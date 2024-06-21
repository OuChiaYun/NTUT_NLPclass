- html檔案一定要放在 'templates' 資料夾裡面 ('render_template()')
- 這個檔案的執行方式就跟普通的python一樣，最主要是要放以下幾行程式碼，才能直接執行，不須另外執行:
    - 'app = Flask(__name__)'
    - 'if __name__ == '__main__':
            app.run()'

- /form
    - 一開始的畫面，表單

- /submit
    - 在Form.html繳交from的action
    - 使用 POST method
    - 在from裡面，我把輸入的文字取名為"INPUT" => request.form['INPUT']
    - redirect(url_for('success', INPUT=INPUT, action="post")) => 把INPUT和action傳到success function

- /success/<action>/<INPUT>
    - 網址會長上面那樣，例如我輸入'hi'，網址就會是'.../success/post/hi'
    - 看要在按按鈕後顯示什麼東西在網頁上