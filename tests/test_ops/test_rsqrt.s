
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n33>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001608: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001614: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s0, s6, s2                                       // 00000000161C: 80000206
	s_addc_u32 s1, s7, s3                                      // 000000001620: 82010307
	s_load_b32 s0, s[0:1], null                                // 000000001624: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_div_scale_f32 v0, null, s0, s0, 1.0                      // 000000001630: D6FC7C00 03C80000
	v_div_scale_f32 v3, vcc_lo, 1.0, s0, 1.0                   // 000000001638: D6FC6A03 03C800F2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001640: BF8700B2
	v_rcp_f32_e32 v1, v0                                       // 000000001644: 7E025500
	s_waitcnt_depctr 0xfff                                     // 000000001648: BF880FFF
	v_fma_f32 v2, -v0, v1, 1.0                                 // 00000000164C: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 000000001654: 56020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001658: BF870091
	v_mul_f32_e32 v2, v3, v1                                   // 00000000165C: 10040303
	v_fma_f32 v4, -v0, v2, v3                                  // 000000001660: D6130004 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001668: BF870091
	v_fmac_f32_e32 v2, v4, v1                                  // 00000000166C: 56040304
	v_fma_f32 v0, -v0, v2, v3                                  // 000000001670: D6130000 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001678: BF870091
	v_div_fmas_f32 v0, v0, v1, v2                              // 00000000167C: D6370000 040A0300
	v_div_fixup_f32 v0, v0, s0, 1.0                            // 000000001684: D6270000 03C80100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000168C: BF8700A1
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001690: 7C2800FF 0F800000
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, vcc_lo              // 000000001698: D5010001 01A9FEF2 4F800000
	v_mul_f32_e32 v0, v0, v1                                   // 0000000016A4: 10000300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000016A8: BF870141
	v_sqrt_f32_e32 v1, v0                                      // 0000000016AC: 7E026700
	s_waitcnt_depctr 0xfff                                     // 0000000016B0: BF880FFF
	v_add_nc_u32_e32 v2, -1, v1                                // 0000000016B4: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 0000000016B8: 4A060281
	v_fma_f32 v4, -v2, v1, v0                                  // 0000000016BC: D6130004 24020302
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016C4: BF870112
	v_fma_f32 v5, -v3, v1, v0                                  // 0000000016C8: D6130005 24020303
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 0000000016D0: D4160000 00020880
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000016D8: BF870221
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 0000000016DC: D5010001 00020501
	v_cndmask_b32_e64 v2, 1.0, 0x37800000, vcc_lo              // 0000000016E4: D5010002 01A9FEF2 37800000
	v_cmp_lt_f32_e32 vcc_lo, 0, v5                             // 0000000016F0: 7C220A80
	s_add_u32 s0, s4, s2                                       // 0000000016F4: 80000204
	s_addc_u32 s1, s5, s3                                      // 0000000016F8: 82010305
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo                       // 0000000016FC: 02020701
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001700: D47E006A 0001FF00 00000260
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000170C: BF870092
	v_mul_f32_e32 v1, v2, v1                                   // 000000001710: 10020302
	v_dual_cndmask_b32 v0, v1, v0 :: v_dual_mov_b32 v1, 0      // 000000001714: CA500101 00000080
	global_store_b32 v1, v0, s[0:1]                            // 00000000171C: DC6A0000 00000001
	s_nop 0                                                    // 000000001724: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001728: BFB60003
	s_endpgm                                                   // 00000000172C: BFB00000
