import{s as v,_ as g}from"./nrsr_share_image-a1d0fe75.mjs";import{_ as m,o as s,c as r,a as e,t as c,F as y,r as $,d as a,f as k,p as b,g as x,h as P,i as I,j as S,u as j,b as p}from"./entry-830e4360.mjs";var w=[{name:"Pellegrini Peter",perc:60.1,perc_floor:60,perc_tens:1,family_name:"Pellegrini",other_names:"Peter"},{name:"Kor\u010Dok Ivan",perc:35.6,perc_floor:35,perc_tens:6,family_name:"Kor\u010Dok",other_names:"Ivan"},{name:"Kubi\u0161 J\xE1n",perc:4.4,perc_floor:4,perc_tens:4,family_name:"Kubi\u0161",other_names:"J\xE1n"}],D=[{date:"2023-12-12"}];const V={data:function(){return{candidates:w,date:new Date(D[0].date).toLocaleDateString("sk")}}},B=a(" \u0160ance na v\xFDhru ve volb\xE1ch "),C=a("dle s\xE1zkov\xFDch kancel\xE1\u0159\xED "),E={class:"fs-5 fw-bolder"},F={class:"container"},K={class:"row"},L={class:"card text-center mb-2"},N={class:"card-header card-title"},z={class:"card-body"},O={class:"number fs-2"},A={key:0},J=a(" %");function M(t,_,n,i,d,u){return s(),r("div",null,[e("h3",null,[B,e("small",null,[C,e("span",E,c(t.date),1)])]),e("div",F,[e("div",K,[(s(!0),r(y,null,$(t.candidates,(o,l)=>(s(),r("div",{key:l,class:"col-xl-3 col-lg-4 col-sm-6"},[e("div",L,[e("h4",N,[a(c(o.family_name)+" ",1),e("small",null,c(o.other_names),1)]),e("div",z,[e("div",O,[a(c(o.perc_floor),1),o.perc_floor<5?(s(),r("small",A,"."+c(o.perc_tens),1)):k("",!0),J])])])]))),128))])])])}var H=m(V,[["render",M]]);const R={},h=t=>(b("data-v-536f1bf9"),t=t(),x(),t),T={id:"image-wrapper"},U=h(()=>e("h3",null,[a(" V\xFDvoj \u0161ance na prezidenta "),e("small",null,"pod\u013Ea st\xE1vkov\xFDch kancel\xE1ri\xED ")],-1)),q=h(()=>e("img",{src:P,class:"image"},null,-1)),G=h(()=>e("img",{src:I,class:"image-small"},null,-1)),Q=[U,q,G];function W(t,_){return s(),r("div",T,Q)}var X=m(R,[["render",W],["__scopeId","data-v-536f1bf9"]]);const Y=S({name:"index",setup(t,{expose:_}){_();const n="https://mandaty.sk/",i="493242628099686";j({title:"Mand\xE1ty.sk",meta:[{hid:"og:name",property:"og:image",content:n+v.filename},{hid:"og:url",property:"og:url",content:n},{hid:"og:type",property:"og:type",content:"website"},{hid:"og:title",property:"og:title",content:"Mand\xE1ty.sk"},{hid:"og:description",property:"og:description",content:"V\xFDvoj volebn\xFDch modelov pod\u013Ea prieskumov verejnej mienky"},{hid:"fb:app_id",property:"fb:app_id",content:i}],link:[{rel:"icon",type:"image/x-icon",href:n+"favicon.svg"}]});const d={BASE_URL:n,FB_APP_ID:i};return Object.defineProperty(d,"__isScriptSetup",{enumerable:!1,value:!0}),d}}),Z={class:"container"},ee=e("h1",{class:"pt-4"},"Prezident/ka 2024",-1),te=e("hr",null,null,-1);function oe(t,_,n,i,d,u){const o=g,l=H,f=X;return s(),r("div",Z,[p(o),ee,p(l),te,p(f)])}var re=m(Y,[["render",oe]]);export{re as default};
