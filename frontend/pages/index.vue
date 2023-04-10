<template>
  <div class="container">
    
    <ChartLinks />

    <h1 class="pt-4">Národná rada 2020-2023</h1>

    <hr />

    <NrsrPollsSeats />

    <hr />

    <NrsrPollsChart />

    <hr />

    <NrsrPollsPartiesTables />

  </div>
</template>

<script setup lang="ts">

import shareImage from '~/assets/data/nrsr/nrsr_share_image.json'

const BASE_URL = 'https://mandaty.sk/'
const FB_APP_ID = '493242628099686'

useHead({
  title: 'Mandáty.sk',
  meta: [
    {hid: 'og:name', property: 'og:image', content: BASE_URL + shareImage['filename']},
    {hid: 'og:url', property: 'og:url', content: BASE_URL},
    {hid: 'og:type', property: 'og:type', content: 'website'},
    {hid: 'og:title', property: 'og:title', content: "Mandáty.sk"},
    {hid: 'og:description', property: 'og:description', content: "Vývoj volebných modelov podľa prieskumov verejnej mienky"},
    {hid: 'fb:app_id', property: 'fb:app_id', content: FB_APP_ID}
  ],
  link: [
    { rel: 'icon', type: 'image/x-icon', href: BASE_URL + 'favicon.svg' },
  ]

})
</script>