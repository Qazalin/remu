
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_5_312_32n3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s2, s14, 5                                      // 000000001714: 8402850E
	s_mulk_i32 s15, 0x2700                                     // 000000001718: B80F2700
	s_add_i32 s2, s2, s13                                      // 00000000171C: 81020D02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001720: BF870499
	s_add_i32 s2, s2, s15                                      // 000000001724: 81020F02
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000172C: BF8704D9
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001734: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001738: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000173C: 82070307
	s_ashr_i32 s15, s14, 31                                    // 000000001740: 860F9F0E
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001744: 8488820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001748: BF870009
	s_add_u32 s0, s0, s8                                       // 00000000174C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001750: 82010901
	s_load_b32 s6, s[6:7], null                                // 000000001754: F4000183 F8000000
	s_load_b32 s0, s[0:1], null                                // 00000000175C: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_div_scale_f32 v0, null, s0, s0, s6                       // 000000001768: D6FC7C00 00180000
	v_div_scale_f32 v3, vcc_lo, s6, s0, s6                     // 000000001770: D6FC6A03 00180006
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001778: BF8700B2
	v_rcp_f32_e32 v1, v0                                       // 00000000177C: 7E025500
	s_waitcnt_depctr 0xfff                                     // 000000001780: BF880FFF
	v_fma_f32 v2, -v0, v1, 1.0                                 // 000000001784: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 00000000178C: 56020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870091
	v_mul_f32_e32 v2, v3, v1                                   // 000000001794: 10040303
	v_fma_f32 v4, -v0, v2, v3                                  // 000000001798: D6130004 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017A0: BF870091
	v_fmac_f32_e32 v2, v4, v1                                  // 0000000017A4: 56040304
	v_fma_f32 v0, -v0, v2, v3                                  // 0000000017A8: D6130000 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000017B0: BF870121
	v_div_fmas_f32 v0, v0, v1, v2                              // 0000000017B4: D6370000 040A0300
	v_mov_b32_e32 v1, 0                                        // 0000000017BC: 7E020280
	v_div_fixup_f32 v0, v0, s0, s6                             // 0000000017C0: D6270000 00180100
	s_add_u32 s0, s4, s2                                       // 0000000017C8: 80000204
	s_addc_u32 s1, s5, s3                                      // 0000000017CC: 82010305
	global_store_b32 v1, v0, s[0:1]                            // 0000000017D0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017D8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017DC: BFB60003
	s_endpgm                                                   // 0000000017E0: BFB00000
