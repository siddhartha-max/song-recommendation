{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'RecommendationSystem'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-e031fb99ee34>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mRecommendationSystem\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mrecommend\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mflask\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mFlask\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mredirect\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0murl_for\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mrender_template\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mapp\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mFlask\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m__name__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'RecommendationSystem'"
     ]
    }
   ],
   "source": [
    "from RecommendationSystem import recommend\n",
    "from flask import Flask,redirect,url_for,request,render_template\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/',methods=['GET'])\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/recommendation',methods=['POST'])\n",
    "def recommend():\n",
    "    Songs_name = request.form['Songs']\n",
    "    similar_Songs = list(recommend(Songs_name.lower()))\n",
    "    print(similar_Songs)\n",
    "    return render_template('recommendation.html',similar_Songs=similar_Songs,recommend=True)\n",
    " \n",
    "\n",
    "if __name__=='__main__':\n",
    "    app.run(debug=True, use_reloader=False)        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named '_main_'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-5-931183804bcf>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0m_main_\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mrecommend\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named '_main_'"
     ]
    }
   ],
   "source": [
    "from _main_ import recommend"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask,redirect,url_for,request,render_template\n",
    "\n",
    "app = Flask(__name__)\n",
    "from werkzeug.middleware.proxy_fix import ProxyFix\n",
    "app.wsgi_app = ProxyFix(app.wsgi_app)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<werkzeug.middleware.proxy_fix.ProxyFix at 0x13dc8c53c40>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "app.wsgi_app"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
