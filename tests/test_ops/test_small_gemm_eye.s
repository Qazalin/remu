
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_8_8_8n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[8:11], s[0:1], null                          // 000000001704: F4080200 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s2, s15, 3                                      // 000000001714: 8402830F
	v_mov_b32_e32 v1, 0                                        // 000000001718: 7E020280
	s_ashr_i32 s3, s2, 31                                      // 00000000171C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[12:13], s[2:3], 2                             // 000000001724: 848C8202
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s2, s10, s12                                     // 00000000172C: 80020C0A
	s_addc_u32 s3, s11, s13                                    // 000000001730: 82030D0B
	s_ashr_i32 s15, s14, 31                                    // 000000001734: 860F9F0E
	s_lshl_b64 s[10:11], s[14:15], 2                           // 000000001738: 848A820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000173C: BF870009
	s_add_u32 s14, s0, s10                                     // 000000001740: 800E0A00
	s_addc_u32 s15, s1, s11                                    // 000000001744: 820F0B01
	s_load_b256 s[0:7], s[2:3], null                           // 000000001748: F40C0001 F8000000
	s_clause 0x7                                               // 000000001750: BF850007
	s_load_b32 s16, s[14:15], null                             // 000000001754: F4000407 F8000000
	s_load_b32 s17, s[14:15], 0x20                             // 00000000175C: F4000447 F8000020
	s_load_b32 s18, s[14:15], 0x40                             // 000000001764: F4000487 F8000040
	s_load_b32 s19, s[14:15], 0x60                             // 00000000176C: F40004C7 F8000060
	s_load_b32 s20, s[14:15], 0x80                             // 000000001774: F4000507 F8000080
	s_load_b32 s21, s[14:15], 0xa0                             // 00000000177C: F4000547 F80000A0
	s_load_b32 s22, s[14:15], 0xc0                             // 000000001784: F4000587 F80000C0
	s_load_b32 s14, s[14:15], 0xe0                             // 00000000178C: F4000387 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 000000001794: BF89FC07
	v_fma_f32 v0, s0, s16, 0                                   // 000000001798: D6130000 02002000
	s_add_u32 s0, s8, s12                                      // 0000000017A0: 80000C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017A4: BF8700C1
	v_fmac_f32_e64 v0, s1, s17                                 // 0000000017A8: D52B0000 00002201
	s_addc_u32 s1, s9, s13                                     // 0000000017B0: 82010D09
	s_add_u32 s0, s0, s10                                      // 0000000017B4: 80000A00
	s_addc_u32 s1, s1, s11                                     // 0000000017B8: 82010B01
	v_fmac_f32_e64 v0, s2, s18                                 // 0000000017BC: D52B0000 00002402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_fmac_f32_e64 v0, s3, s19                                 // 0000000017C8: D52B0000 00002603
	v_fmac_f32_e64 v0, s4, s20                                 // 0000000017D0: D52B0000 00002804
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_fmac_f32_e64 v0, s5, s21                                 // 0000000017DC: D52B0000 00002A05
	v_fmac_f32_e64 v0, s6, s22                                 // 0000000017E4: D52B0000 00002C06
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017EC: BF870001
	v_fmac_f32_e64 v0, s7, s14                                 // 0000000017F0: D52B0000 00001C07
	global_store_b32 v1, v0, s[0:1]                            // 0000000017F8: DC6A0000 00000001
	s_nop 0                                                    // 000000001800: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001804: BFB60003
	s_endpgm                                                   // 000000001808: BFB00000
