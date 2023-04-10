<template>
  <div id="image-wrapper">
    <h3>
      Vývoj volebných modelov 
      <small>podľa prieskumov verejnej mienky
      </small>
    </h3>
    <div v-if="isMobile">
      <NrsrPollsPictureChart />
    </div>
    <div v-if="!isMobile">
      <NrsrPollsInteractiveChart />
    </div>
  </div>
</template>

<script>

export default {
  name: 'nRSRPollsChart',
  data() {
    return {
      isMobile: false
    }
  },
  mounted() {
    this.isMobile = window.innerWidth < 666
    console.log('isMobile:', this.isMobile)
  },
  components: {
    NrsrPollsPictureChart: () => import('~/components/nrsr/NrsrPollsPictureChart.vue'),
    NrsrPollsInteractiveChart: () => import('~/components/nrsr/NrsrPollsInteractiveChart.vue')
  }
}
</script>

<style scoped>
#image-wrapper {
  min-height: 40vh;
}
</style>